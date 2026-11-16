# coding=utf-8
"""首页：按传感器类别展示卡片，点击进入对应类别的查看器。

类似 vitai-hand-data-viz 的 home 页：每种传感器类别一张卡片，卡片上展示
特性说明与「进入查看器」按钮；点击后通过 category_selected 信号通知主窗口
切换到该类别的查看器页面（查看器内的设备列表只含当前类别的传感器）。

首页会轻量扫描一次设备，统计每个类别已接入的传感器数量，用于提示与按钮可用态。
"""

from pathlib import Path

from PyQt6.QtCore import pyqtSignal, QTimer, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
)
from PyQt6.QtGui import QFont, QPixmap

from app.sensor_categories import SENSOR_CATEGORIES, category_for_sn
from app.widgets.device_panel import DeviceScanWorker

# 静态资源目录（app/images），基于本文件定位，打包后亦可用
IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"


class _CategoryCard(QFrame):
    """单个类别卡片：图标徽标 + 标题 + 副标题 + 特性列表 + 进入按钮。"""

    selected = pyqtSignal(str)

    def __init__(self, category, parent=None):
        super().__init__(parent)
        self.setObjectName("CategoryCard")
        self._cat = category
        self._setup_ui()

    def _setup_ui(self):
        c = self._cat
        v = QVBoxLayout(self)
        v.setContentsMargins(28, 28, 28, 28)
        v.setSpacing(14)
        v.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 产品图（hand.png / 225.png），缺失时退回文字徽标
        image_path = IMAGES_DIR / c["image"] if c.get("image") else None
        if image_path and image_path.exists():
            pix = QPixmap(str(image_path))
            if not pix.isNull():
                img = QLabel()
                img.setObjectName("CardImage")
                img.setPixmap(
                    pix.scaled(
                        160, 160,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                img.setFixedSize(180, 180)
                img.setAlignment(Qt.AlignmentFlag.AlignCenter)
                v.addWidget(img, 0, Qt.AlignmentFlag.AlignHCenter)
            else:
                self._add_badge(v, c)
        else:
            self._add_badge(v, c)

        # 标题 + 副标题
        title = QLabel(c["title"])
        title.setObjectName("CardTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(title)

        sub = QLabel(c["subtitle"])
        sub.setObjectName("CardSubtitle")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(sub)

        # 特性列表（✓ 前缀）
        for feat in c["features"]:
            item = QLabel(f"✓  {feat}")
            item.setObjectName("CardFeature")
            v.addWidget(item)

        v.addStretch(1)

        # 设备计数提示（扫描后更新）
        self._count_label = QLabel("检测中…")
        self._count_label.setObjectName("CardCount")
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._count_label)

        # 进入按钮（渐变）
        self._enter_btn = QPushButton(c["button"])
        self._enter_btn.setObjectName("EnterBtn")
        self._enter_btn.setProperty("catColor", c["color"])
        self._enter_btn.clicked.connect(
            lambda: self.selected.emit(c["id"])
        )
        v.addWidget(self._enter_btn)

    def _add_badge(self, v, c):
        """图片缺失时的降级：圆形色块 + 类别短字。"""
        badge = QLabel(c["badge"])
        badge.setObjectName("CardBadge")
        badge.setFixedSize(72, 72)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFont(QFont("", 30, QFont.Weight.Bold))
        badge.setStyleSheet(
            f"background-color: {c['color']}; color: #0a0f1a; "
            "border-radius: 36px;"
        )
        v.addWidget(badge, 0, Qt.AlignmentFlag.AlignHCenter)

    def category_id(self):
        return self._cat["id"]

    def set_device_count(self, n):
        """更新设备计数提示与按钮可用态。

        n 为 None 表示尚未扫描完成（按钮可用，提示「检测中…」）；
        n == 0 表示未检测到该类别设备（按钮禁用）；否则显示数量。
        """
        if n is None:
            self._count_label.setText("检测中…")
            self._enter_btn.setEnabled(True)
            self._enter_btn.setProperty("disabled", False)
        elif n == 0:
            self._count_label.setText("未检测到该类别设备")
            self._enter_btn.setEnabled(False)
            self._enter_btn.setProperty("disabled", True)
        else:
            self._count_label.setText(f"检测到 {n} 个传感器")
            self._enter_btn.setEnabled(True)
            self._enter_btn.setProperty("disabled", False)
        # 触发样式刷新（disabled 状态）
        self._enter_btn.style().unpolish(self._enter_btn)
        self._enter_btn.style().polish(self._enter_btn)


class HomePage(QWidget):
    category_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HomePage")
        self._scan_worker = None
        self._cards = []
        # 实时出流 SN 提供者：出流中的传感器被 SDK 独占，扫描不到，用于计数兜底
        self._live_sns_provider = None
        self._setup_ui()

        self._scan_timer = QTimer(self)
        self._scan_timer.setInterval(3000)
        self._scan_timer.timeout.connect(self._start_scan)

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 顶部 logo（logo5.png）+ 标题
        header = QHBoxLayout()
        header.setContentsMargins(28, 22, 28, 0)
        header.setSpacing(12)
        logo_path = IMAGES_DIR / "logo5.png"
        logo_pix = QPixmap(str(logo_path))
        if not logo_pix.isNull():
            logo = QLabel()
            logo.setObjectName("HomeLogoImage")
            logo.setPixmap(
                logo_pix.scaledToHeight(
                    44, Qt.TransformationMode.SmoothTransformation
                )
            )
            logo.setFixedHeight(52)
        else:
            logo = QLabel("ViTai")
            logo.setObjectName("HomeLogo")
        header.addWidget(logo)
        header.addStretch(1)
        root.addLayout(header)

        title = QLabel("视触觉传感器查看器")
        title.setObjectName("HomeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        sub = QLabel("请选择传感器类别")
        sub.setObjectName("HomeSubtitle")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(sub)

        root.addSpacing(28)

        # 卡片区：水平居中排列
        cards_row = QHBoxLayout()
        cards_row.setContentsMargins(0, 0, 0, 0)
        cards_row.setSpacing(36)
        cards_row.addStretch(1)
        for c in SENSOR_CATEGORIES:
            card = _CategoryCard(c, self)
            self._cards.append(card)
            card.selected.connect(self.category_selected.emit)
            cards_row.addWidget(card, 0, Qt.AlignmentFlag.AlignTop)
        cards_row.addStretch(1)
        root.addLayout(cards_row)

        root.addStretch(1)

    def _start_scan(self):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            return
        worker = DeviceScanWorker()
        worker.devices_found.connect(self._on_devices)
        worker.finished.connect(self._on_scan_finished)
        self._scan_worker = worker
        worker.start()

    def _on_scan_finished(self):
        # 等线程真正结束后再释放引用，避免 QThread 在 run() 未退出时被回收
        worker = self._scan_worker
        self._scan_worker = None
        if worker is not None:
            worker.deleteLater()

    def set_live_sns_provider(self, provider):
        """注册「正在出流（独占相机）的 SN」提供者，供扫描结果兜底。"""
        self._live_sns_provider = provider

    def refresh_devices(self):
        """重新扫描设备（返回首页时调用）。"""
        for card in self._cards:
            card.set_device_count(None)
        if self.isVisible():
            self._start_scan()

    def _on_devices(self, sns):
        # 扫描结果与「正在出流被独占」的 SN 取并集，避免误报未检测到
        all_sns = set(sns or ())
        if self._live_sns_provider is not None:
            try:
                all_sns |= {s for s in self._live_sns_provider() if s}
            except Exception:
                pass
        counts = {}
        for s in all_sns:
            c = category_for_sn(s)
            if c:
                counts[c] = counts.get(c, 0) + 1
        for card in self._cards:
            card.set_device_count(
                counts.get(card.category_id(), 0))

    def showEvent(self, ev):
        super().showEvent(ev)
        self._scan_timer.start()
        self._start_scan()

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self._scan_timer.stop()
