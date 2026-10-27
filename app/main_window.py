from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QStatusBar,
    QLabel,
    QMessageBox,
)
from PyQt6.QtCore import Qt

from app.build_info import APP_VERSION, GIT_BRANCH, SDK_WHEEL
from app.sensor_worker import SensorWorker
from pyvitaisdk import VTSDataType


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 原生系统窗口标题
        self.setWindowTitle(
            f"ViTai 视触觉传感器查看器 | Version: {APP_VERSION}"
        )

        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)

        self._workers = {}   # sn -> SensorWorker
        self._configs = []   # 已连接的传感器配置
        self._fps = {}       # sn -> 最近帧率

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        # 局部导入，便于 UI 热更新后拿到 reload 过的类
        from app.widgets.device_panel import DevicePanel
        from app.widgets.multi_sensor_viewer import MultiSensorViewer

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # =========================================================
        # 顶部构建信息栏
        #
        # 左：应用版本
        # 中：应用名称
        # 右：SDK wheel
        #
        # 三列等宽，确保中间标题相对于整个窗口真正居中。
        # =========================================================
        self.build_info_bar = QWidget()
        self.build_info_bar.setObjectName("BuildInfoBar")
        self.build_info_bar.setFixedHeight(40)

        build_info_layout = QGridLayout(self.build_info_bar)
        build_info_layout.setContentsMargins(16, 0, 16, 0)
        build_info_layout.setHorizontalSpacing(8)
        build_info_layout.setVerticalSpacing(0)

        # 左侧：应用版本
        self.version_label = QLabel(
            f"Version: {APP_VERSION}"
        )
        self.version_label.setObjectName("VersionLabel")
        self.version_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
        )

        # Tooltip 中保留完整构建来源信息，方便调试。
        self.version_label.setToolTip(
            f"Version: {APP_VERSION}\n"
            f"Git branch: {GIT_BRANCH}"
        )

        # 中间：应用标题
        self.title_label = QLabel(
            "ViTai 视触觉传感器查看器"
        )
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        # 右侧：SDK wheel
        self.sdk_wheel_label = QLabel(
            SDK_WHEEL
        )
        self.sdk_wheel_label.setObjectName("SdkWheelLabel")
        self.sdk_wheel_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.sdk_wheel_label.setToolTip(
            f"SDK wheel: {SDK_WHEEL}"
        )

        # 三列布局
        build_info_layout.addWidget(
            self.version_label,
            0,
            0,
        )

        build_info_layout.addWidget(
            self.title_label,
            0,
            1,
        )

        build_info_layout.addWidget(
            self.sdk_wheel_label,
            0,
            2,
        )

        # 三列等宽。
        #
        # 即使 SDK_WHEEL 比 Version 文本长很多，
        # 中间标题仍然保持在窗口的几何中心。
        build_info_layout.setColumnStretch(0, 1)
        build_info_layout.setColumnStretch(1, 1)
        build_info_layout.setColumnStretch(2, 1)

        main_layout.addWidget(
            self.build_info_bar
        )

        # =========================================================
        # 顶部 - 设备控制栏
        # =========================================================
        self.device_panel = DevicePanel()
        main_layout.addWidget(
            self.device_panel
        )

        # =========================================================
        # 中部 - 多传感器滚动显示区
        # =========================================================
        self.multi_viewer = MultiSensorViewer()
        main_layout.addWidget(
            self.multi_viewer,
            1,
        )

        # =========================================================
        # 状态栏
        # =========================================================
        self.status_bar = QStatusBar()
        self.setStatusBar(
            self.status_bar
        )

        self.status_label = QLabel("就绪")
        self.fps_label = QLabel("帧率: --")
        self.ts_label = QLabel("时间戳: --")

        self.status_bar.addWidget(
            self.status_label
        )

        self.status_bar.addPermanentWidget(
            self.fps_label
        )

        self.status_bar.addPermanentWidget(
            self.ts_label
        )

    def _connect_signals(self):
        self.device_panel.connect_request.connect(
            self._on_connect
        )
        self.device_panel.disconnect_request.connect(
            self._on_disconnect
        )
        self.device_panel.start_request.connect(
            self._on_start
        )
        self.device_panel.stop_request.connect(
            self._on_stop
        )
        self.device_panel.calibrate_request.connect(
            self._on_calibrate
        )

    def _disconnect_panel_signals(self):
        panel = getattr(self, "device_panel", None)
        if panel is None:
            return
        for signal in (
            panel.connect_request,
            panel.disconnect_request,
            panel.start_request,
            panel.stop_request,
            panel.calibrate_request,
        ):
            try:
                signal.disconnect()
            except TypeError:
                pass

    def reload_ui(self):
        """就地重建 UI，保留传感器 worker 连接。"""
        from app.ui_hot_reload import reload_widget_modules

        panel_state = None
        viewer_sns = []
        status_text = "就绪"
        fps_text = "帧率: --"
        ts_text = "时间戳: --"

        if hasattr(self, "device_panel"):
            panel_state = self.device_panel.export_ui_state()
            self.device_panel.prepare_destroy()
            self._disconnect_panel_signals()

        if hasattr(self, "multi_viewer"):
            viewer_sns = self.multi_viewer.sensor_sns()

        if hasattr(self, "status_label"):
            status_text = self.status_label.text()
            fps_text = self.fps_label.text()
            ts_text = self.ts_label.text()

        reload_widget_modules()
        self._setup_ui()
        self._connect_signals()

        if panel_state is not None:
            self.device_panel.apply_ui_state(panel_state)

        if viewer_sns:
            self.multi_viewer.set_sensors(viewer_sns)
            if self._workers:
                self.multi_viewer.set_loading(viewer_sns, False)

        self.status_label.setText(status_text)
        self.fps_label.setText(fps_text)
        self.ts_label.setText(ts_text)

        self.status_bar.showMessage("UI 已热更新", 2000)
        print("[reload] UI 已就地刷新（传感器连接保留）")

    def _sns(self):
        return [
            c.SN
            for c in self._configs
        ]

    def _on_connect(self, configs):
        self._configs = list(configs)
        self._start_all_workers()

    def _start_all_workers(self):
        self._stop_all_workers()

        sns = self._sns()

        self.multi_viewer.set_sensors(
            sns
        )
        self.multi_viewer.set_loading(
            sns,
            True,
        )

        self._fps.clear()

        marker_size = (
            self.device_panel.get_marker_size()
        )

        offsets = (
            self.device_panel.get_marker_offsets()
        )

        weight_dir = (
            self.device_panel.get_weight_dir()
        )

        for config in self._configs:
            sn = config.SN

            worker = SensorWorker(
                config,
                marker_size,
                offsets,
                weight_dir,
                display_freq=30,
            )

            worker.data_ready.connect(
                lambda data, sn=sn:
                self._on_data_ready(
                    sn,
                    data,
                )
            )

            worker.error_occurred.connect(
                lambda msg, sug, sn=sn:
                self._on_worker_error(
                    sn,
                    msg,
                    sug,
                )
            )

            worker.fps_updated.connect(
                lambda fps, sn=sn:
                self._on_fps_updated(
                    sn,
                    fps,
                )
            )

            worker.weight_status.connect(
                self._on_weight_status
            )

            self._workers[sn] = worker
            worker.start()

        self.status_label.setText(
            f"采集中 ({len(self._configs)}个传感器)"
        )

    def _on_disconnect(self):
        self._stop_all_workers()

        self._configs = []
        self._fps.clear()

        self.status_label.setText(
            "就绪"
        )
        self.fps_label.setText(
            "帧率: --"
        )
        self.ts_label.setText(
            "时间戳: --"
        )

        self.multi_viewer.reset()

    def _on_start(self):
        if not self._configs:
            return

        self._start_all_workers()

    def _on_stop(self):
        self._stop_all_workers()

        self._fps.clear()

        self.status_label.setText(
            "已停止"
        )
        self.fps_label.setText(
            "帧率: --"
        )
        self.ts_label.setText(
            "时间戳: --"
        )

    def _on_calibrate(self):
        for worker in self._workers.values():
            if worker.isRunning():
                worker.calibrate()

        self.status_bar.showMessage(
            "已重新校准",
            2000,
        )

    def _on_data_ready(
        self,
        sn,
        data,
    ):
        self.multi_viewer.set_loading(
            [sn],
            False,
        )

        self.multi_viewer.update_data(
            sn,
            data,
        )

        if VTSDataType.TIME_STAMP in data:
            self.ts_label.setText(
                f"时间戳: "
                f"{data[VTSDataType.TIME_STAMP]}"
            )

    def _on_weight_status(
        self,
        applied,
        msg,
    ):
        self.status_bar.showMessage(
            msg,
            3000,
        )

    def _on_fps_updated(
        self,
        sn,
        fps,
    ):
        self._fps[sn] = fps

        if self._fps:
            avg = (
                sum(self._fps.values())
                / len(self._fps)
            )

            self.fps_label.setText(
                f"帧率: {avg:.1f}"
            )

    def _on_worker_error(
        self,
        sn,
        message,
        suggestion,
    ):
        self._remove_worker(sn)

        self.multi_viewer.remove_sensor(
            sn
        )

        if self._workers:
            self.status_bar.showMessage(
                f"[{sn}] {message}",
                5000,
            )
        else:
            full = message

            if suggestion:
                full += (
                    f"\n\n建议: {suggestion}"
                )

            QMessageBox.critical(
                self,
                f"传感器错误 ({sn})",
                full,
            )

            self.status_label.setText(
                f"错误: {message[:60]}"
            )

            self.device_panel.on_worker_disconnected()

    def _remove_worker(
        self,
        sn,
    ):
        worker = self._workers.pop(
            sn,
            None,
        )

        if worker is not None:
            if worker.isRunning():
                worker.stop()
                worker.wait(3000)

            if worker.isRunning():
                worker.terminate()
                worker.wait()

    def _stop_all_workers(self):
        for sn in list(
            self._workers.keys()
        ):
            self._remove_worker(sn)

        self._workers.clear()

    def closeEvent(
        self,
        event,
    ):
        self._stop_all_workers()
        event.accept()