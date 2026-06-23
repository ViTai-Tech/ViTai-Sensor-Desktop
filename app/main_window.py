from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStatusBar, QLabel, QMessageBox
from PyQt6.QtCore import Qt
from app.widgets.device_panel import DevicePanel
from app.widgets.image_viewer import ImageViewer
from app.widgets.data_panel import DataPanel
from app.sensor_worker import SensorWorker
from pyvitaisdk import VTSDataType


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ViTai 视触觉传感器查看器")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)

        self._worker = None
        self._config = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top - Device control bar
        self.device_panel = DevicePanel()
        main_layout.addWidget(self.device_panel)

        sep_top = QWidget()
        sep_top.setFixedHeight(1)
        sep_top.setStyleSheet("background-color: #cccccc;")
        main_layout.addWidget(sep_top)

        # Bottom - Image + Data
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        self.image_viewer = ImageViewer()
        bottom_layout.addWidget(self.image_viewer, 3)

        sep_mid = QWidget()
        sep_mid.setFixedWidth(1)
        sep_mid.setStyleSheet("background-color: #cccccc;")
        bottom_layout.addWidget(sep_mid)

        self.data_panel = DataPanel()
        bottom_layout.addWidget(self.data_panel, 2)

        main_layout.addLayout(bottom_layout, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("就绪")
        self.fps_label = QLabel("帧率: --")
        self.ts_label = QLabel("时间戳: --")
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.fps_label)
        self.status_bar.addPermanentWidget(self.ts_label)

    def _connect_signals(self):
        self.device_panel.connect_request.connect(self._on_connect)
        self.device_panel.disconnect_request.connect(self._on_disconnect)
        self.device_panel.start_request.connect(self._on_start)
        self.device_panel.stop_request.connect(self._on_stop)
        self.device_panel.calibrate_request.connect(self._on_calibrate)

    def _on_connect(self, config):
        self._config = config
        self.device_panel._on_start()

    def _on_disconnect(self):
        self._stop_worker()
        self._config = None
        self.status_label.setText("就绪")
        self.fps_label.setText("帧率: --")
        self.ts_label.setText("时间戳: --")
        self.data_panel.reset()

    def _on_start(self):
        if self._config is None:
            return

        marker_size = self.device_panel.get_marker_size()
        offsets = self.device_panel.get_marker_offsets()
        weight_dir = self.device_panel.get_weight_dir()

        self._worker = SensorWorker(self._config, marker_size, offsets, weight_dir,
                                    display_freq=30)
        self._worker.data_ready.connect(self._on_data_ready)
        self._worker.sensor_info.connect(self._on_sensor_info)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.fps_updated.connect(self._on_fps_updated)
        self._worker.weight_status.connect(self._on_weight_status)
        self._worker.connected.connect(self.device_panel.on_worker_connected)
        self._worker.disconnected.connect(self.device_panel.on_worker_disconnected)
        self._worker.start()

        self.data_panel.reset()
        self.image_viewer.set_loading(True)
        self.status_label.setText("初始化中...")

    def _on_stop(self):
        self._stop_worker()
        self.status_label.setText("已停止")
        self.fps_label.setText("帧率: --")
        self.ts_label.setText("时间戳: --")
        self.device_panel.on_worker_disconnected()

    def _on_calibrate(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.calibrate()
            self.status_bar.showMessage("已重新校准", 2000)

    def _on_data_ready(self, data):
        self.image_viewer.set_loading(False)
        self.image_viewer.update_data(data)
        self.data_panel.update_data(data)

        if VTSDataType.TIME_STAMP in data:
            self.ts_label.setText(f"时间戳: {data[VTSDataType.TIME_STAMP]}")

    def _on_sensor_info(self, sensor_type):
        self.status_label.setText(f"采集中 ({sensor_type})")

    def _on_weight_status(self, applied, msg):
        self.status_bar.showMessage(msg, 3000)

    def _on_fps_updated(self, fps):
        self.fps_label.setText(f"帧率: {fps:.1f}")

    def _on_worker_error(self, message, suggestion):
        self._stop_worker()
        msg = message
        if suggestion:
            msg += f"\n\n建议: {suggestion}"
        QMessageBox.critical(self, "传感器错误", msg)
        self.status_label.setText(f"错误: {message[:60]}")
        self.device_panel.on_worker_disconnected()

    def _stop_worker(self):
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.stop()
                self._worker.wait(3000)
            if self._worker.isRunning():
                self._worker.terminate()
                self._worker.wait()
            self._worker = None

    def closeEvent(self, event):
        self._stop_worker()
        event.accept()
