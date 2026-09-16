from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QFrame,
    QStackedWidget,
    QScrollBar,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from app.widgets.image_viewer import ImageLabel
from app.widgets.data_panel import ForcePlots
from pyvitaisdk import VTSDataType


class SensorRowWidget(QFrame):
    """单个传感器的显示行：名称 + 原图(力叠加在左上角) + 深度图 + 标记点 + 折线图，全部在同一行。"""

    def __init__(self, sn, parent=None):
        super().__init__(parent)
        self.sn = sn
        self.setObjectName("SensorRow")
        self._setup_ui()

    def _setup_ui(self):
        # 整体行：右上角放小号滑动状态，下面为名称 + 图像 + 波形
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 2, 6, 4)
        v.setSpacing(2)

        self.slip_label = QLabel("● --")
        self.slip_label.setFont(QFont("", 9, QFont.Weight.Bold))
        self.slip_label.setStyleSheet("color: #2ecc71;")
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(0)
        header.addStretch(1)
        header.addWidget(self.slip_label)
        v.addLayout(header)

        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        # 传感器名称（行首）
        name = QLabel(self.sn)
        name.setFixedWidth(96)
        name.setWordWrap(True)
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setStyleSheet("font-weight: bold; color: #333; font-size: 12px;")
        h.addWidget(name)

        # 原图
        self.raw_view = ImageLabel()
        raw_tile = self._tile("原图", self.raw_view)

        # 深度图
        self.depth_view = ImageLabel()
        depth_tile = self._tile("深度图", self.depth_view)

        # 标记点
        self.marker_view = ImageLabel()
        marker_tile = self._tile("标记点", self.marker_view)

        # 折线图（三根独立，竖排一列）：去掉右下角滑动状态后占满整列，随行高伸缩
        self.force_plots = ForcePlots()
        for p in (self.force_plots.fx_plot, self.force_plots.fy_plot, self.force_plots.fz_plot):
            p.setMinimumHeight(40)

        h.addWidget(raw_tile, 3)
        h.addWidget(depth_tile, 3)
        h.addWidget(marker_tile, 3)
        h.addWidget(self.force_plots, 4)

        v.addLayout(h, 1)

    def _tile(self, title, view, overlay=None):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(2)
        cap = QLabel(title)
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.addWidget(cap)
        if overlay is not None:
            body = QWidget()
            g = QGridLayout(body)
            g.setContentsMargins(0, 0, 0, 0)
            g.setSpacing(0)
            g.addWidget(view, 0, 0)
            g.addWidget(overlay, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            # 让图像填满 body，而不是被 Ignored 尺寸策略压到最小尺寸
            g.setColumnStretch(0, 1)
            g.setRowStretch(0, 1)
            l.addWidget(body, 1)
        else:
            l.addWidget(view, 1)
        return w

    def set_loading(self, loading):
        for view in (self.raw_view, self.depth_view, self.marker_view):
            view.set_loading(loading)

    def update_data(self, data):
        if VTSDataType.WARPED_IMG in data:
            self.raw_view.set_image(data[VTSDataType.WARPED_IMG])
        if VTSDataType.DEPTH_MAP in data:
            self.depth_view.set_image(data[VTSDataType.DEPTH_MAP], is_depth=True)
        if VTSDataType.MARKER_IMG in data:
            self.marker_view.set_image(data[VTSDataType.MARKER_IMG])
        if VTSDataType.FORCE6D_VECTOR in data:
            f = data[VTSDataType.FORCE6D_VECTOR]
            if f is not None and len(f) >= 6:
                self.force_plots.update_data(f)
        if VTSDataType.SLIP_STATE in data:
            self._set_slip(data[VTSDataType.SLIP_STATE])

    def _set_slip(self, state):
        if state is None:
            return
        name = state.name if hasattr(state, "name") else str(state)
        if "SLIP" in name.upper():
            self.slip_label.setStyleSheet("color: #e74c3c;")
        else:
            self.slip_label.setStyleSheet("color: #2ecc71;")
        self.slip_label.setText(f"● {name}")


def _chunk(seq, n):
    """把序列切成若干长度不超过 n 的小块。"""
    return [seq[i:i + n] for i in range(0, len(seq), n)]


class MultiSensorViewer(QWidget):
    """按页显示传感器：每页最多 PAGE_SIZE 个，滚轮或右侧滑块翻页。"""

    PAGE_SIZE = 5  # 单页最多显示的传感器数量

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = {}   # sn -> SensorRowWidget
        self._sns = []    # 当前传感器序列号（有序，用于分页）

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._stack = QStackedWidget()
        self._scroll = QScrollBar(Qt.Orientation.Vertical)
        self._scroll.setSingleStep(1)
        self._scroll.setPageStep(1)
        self._scroll.valueChanged.connect(self._stack.setCurrentIndex)

        body.addWidget(self._stack, 1)
        body.addWidget(self._scroll)
        outer.addLayout(body)

    def wheelEvent(self, ev):
        # 滚轮翻页：下滑下一页，上滑上一页
        if self._scroll.maximum() <= 0:
            super().wheelEvent(ev)
            return
        dy = ev.angleDelta().y()
        if dy < 0:
            self._scroll.setValue(self._scroll.value() + 1)
        elif dy > 0:
            self._scroll.setValue(self._scroll.value() - 1)
        else:
            super().wheelEvent(ev)
        ev.accept()

    def set_sensors(self, sns):
        self._sns = list(sns)
        self._rebuild()

    def remove_sensor(self, sn):
        if sn in self._sns:
            self._sns.remove(sn)
        self._rebuild()

    def _rebuild(self):
        # 清空旧页面与行
        while self._stack.count() > 0:
            w = self._stack.widget(0)
            self._stack.removeWidget(w)
            w.deleteLater()
        self._rows.clear()

        # 每页最多 PAGE_SIZE 个，不足时行会自动拉伸填满页面
        for page_sns in _chunk(self._sns, self.PAGE_SIZE):
            page = QWidget()
            pl = QVBoxLayout(page)
            pl.setContentsMargins(8, 8, 8, 8)
            pl.setSpacing(8)
            for sn in page_sns:
                row = SensorRowWidget(sn)
                self._rows[sn] = row
                pl.addWidget(row, 1)
            self._stack.addWidget(page)

        n = self._stack.count()
        self._scroll.setRange(0, max(0, n - 1))
        self._scroll.setVisible(n > 1)
        self._scroll.setValue(0)

    def update_data(self, sn, data):
        row = self._rows.get(sn)
        if row is not None:
            row.update_data(data)

    def set_loading(self, sns, loading):
        for sn in sns:
            row = self._rows.get(sn)
            if row is not None:
                row.set_loading(loading)

    def reset(self):
        self._sns = []
        self._rebuild()

    def count(self):
        return len(self._rows)
