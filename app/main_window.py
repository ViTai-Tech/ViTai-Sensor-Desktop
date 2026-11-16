from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QStatusBar,
    QStackedWidget,
)

from app.build_info import APP_VERSION, GIT_BRANCH, SDK_WHEEL
from app.widgets.home_page import HomePage
from app.widgets.viewer_page import ViewerPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 原生系统窗口标题
        self.setWindowTitle(
            f"ViTai 视触觉传感器查看器 | Version: {APP_VERSION}"
        )

        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)

        # 每个传感器类别对应一个常驻的 ViewerPage（保留各自的连接状态）
        self._viewer_pages = {}

        self._setup_ui()
        self._go_home()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # =========================================================
        # 内容区：首页 + 各类别查看器页（用 QStackedWidget 切换）
        # =========================================================
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        self.home_page = HomePage()
        self.stack.addWidget(self.home_page)
        self.home_page.category_selected.connect(self._open_category)
        # 首页设备计数兜底：正在出流的传感器会被 SDK 独占而扫描不到
        self.home_page.set_live_sns_provider(self._streaming_sns)

        # =========================================================
        # 状态栏
        # =========================================================
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("就绪")
        self.ts_label = QLabel("时间戳: --")

        self.build_info_label = QLabel(
            f"Git: {GIT_BRANCH} | SDK: {SDK_WHEEL}"
        )
        self.build_info_label.setObjectName("BuildInfoLabel")
        self.build_info_label.setToolTip(
            f"Git branch: {GIT_BRANCH}\n"
            f"SDK wheel: {SDK_WHEEL}"
        )

        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.build_info_label)
        self.status_bar.addPermanentWidget(self.ts_label)

    # ------------------------------------------------------------------ #
    # 导航
    # ------------------------------------------------------------------ #
    def _go_home(self):
        # 有正在进行的采集时，先走与「退出数据采集」一致的询问流程；
        # 用户取消则留在当前页面，不做任何改动。
        for page in self._viewer_pages.values():
            try:
                if not page.confirm_leave_capture():
                    return
            except Exception:
                pass
        # 先关掉所有出流，释放相机，否则首页扫描不到被占用的设备
        self._release_devices()
        self.stack.setCurrentWidget(self.home_page)
        self.status_label.setText("就绪")
        # 返回首页时重新扫描一次设备
        self.home_page.refresh_devices()

    def _release_devices(self):
        for page in self._viewer_pages.values():
            try:
                page.release_devices()
            except Exception:
                pass

    def _streaming_sns(self):
        sns = []
        for page in self._viewer_pages.values():
            try:
                sns.extend(page.streaming_sns())
            except Exception:
                pass
        return sns

    def _open_category(self, cat_id):
        page = self._viewer_pages.get(cat_id)
        if page is None:
            page = ViewerPage(cat_id)
            page.back_requested.connect(self._go_home)
            page.status_changed.connect(self.status_label.setText)
            page.timestamp_changed.connect(self.ts_label.setText)
            page.transient_message.connect(
                lambda m, t=2000: self.status_bar.showMessage(m, t)
            )
            self._viewer_pages[cat_id] = page
            self.stack.addWidget(page)

        self.stack.setCurrentWidget(page)
        # 进入页面时立即刷新一次设备列表
        page.on_shown()

    # ------------------------------------------------------------------ #
    # 热更新：首页无状态直接重建；各查看器页保留 worker 连接就地刷新
    # ------------------------------------------------------------------ #
    def reload_ui(self):
        was_home = self.stack.currentWidget() is self.home_page

        # 重建首页
        self.stack.removeWidget(self.home_page)
        self.home_page.deleteLater()
        self.home_page = HomePage()
        self.home_page.category_selected.connect(self._open_category)
        self.home_page.set_live_sns_provider(self._streaming_sns)
        self.stack.insertWidget(0, self.home_page)

        # 各查看器页就地刷新（保留传感器连接）
        for page in self._viewer_pages.values():
            page.reload_ui()

        self.status_bar.showMessage("UI 已热更新", 2000)
        print("[reload] UI 已就地刷新（传感器连接保留）")

        if was_home:
            self.stack.setCurrentWidget(self.home_page)

    def closeEvent(self, event):
        for page in self._viewer_pages.values():
            try:
                page.shutdown()
            except Exception:
                pass
        event.accept()
