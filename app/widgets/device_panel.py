import os
import json

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QGroupBox,
    QSpinBox,
    QGridLayout,
    QMessageBox,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal, QTimer, QThread
from pyvitaisdk import VTSDeviceFinder, VTSError


class DeviceScanWorker(QThread):
    devices_found = pyqtSignal(list)
    scan_error = pyqtSignal(str)

    def run(self):
        try:
            finder = VTSDeviceFinder()
            sns = finder.get_sns()
            self.devices_found.emit(sns)
        except VTSError as e:
            self.scan_error.emit(str(e))
        except Exception as e:
            self.scan_error.emit(str(e))


import sys as _sys


def _get_settings_path():
    if getattr(_sys, "frozen", False):
        d = os.path.dirname(_sys.executable)
    else:
        d = os.path.dirname(os.path.abspath(__file__))
        d = os.path.join(d, "..", "..")
    return os.path.abspath(os.path.join(d, "settings.json"))


SETTINGS_FILE = _get_settings_path()


class DevicePanel(QWidget):
    connect_request = pyqtSignal(object)
    disconnect_request = pyqtSignal()
    start_request = pyqtSignal()
    stop_request = pyqtSignal()
    calibrate_request = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)
        self._connected = False
        self._streaming = False
        self._had_devices = False
        self._scan_worker = None
        self._weight_dir = None
        self._setup_ui()
        self._restore_settings()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._start_scan)
        self._refresh_timer.start(2000)
        self._start_scan()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        sn_layout = QHBoxLayout()
        sn_layout.addWidget(QLabel("序列号:"))
        self.sn_combo = QComboBox()
        self.sn_combo.setMinimumWidth(140)
        sn_layout.addWidget(self.sn_combo)
        layout.addLayout(sn_layout)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self._start_scan)
        layout.addWidget(self.refresh_btn)

        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self._on_connect)
        layout.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("断开")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.clicked.connect(self._on_disconnect)
        layout.addWidget(self.disconnect_btn)

        self.model_label = QLabel("型号: --")
        layout.addWidget(self.model_label)

        layout.addSpacing(16)

        layout.addWidget(QLabel("Marker数量:"))
        self.marker_spin = QSpinBox()
        self.marker_spin.setRange(1, 99)
        self.marker_spin.setValue(12)
        self.marker_spin.setToolTip("设为21以启用六维力估计")
        self.marker_spin.setFixedWidth(60)
        layout.addWidget(self.marker_spin)

        layout.addSpacing(16)

        self.calib_btn = QPushButton("校准")
        self.calib_btn.setEnabled(False)
        self.calib_btn.clicked.connect(self.calibrate_request.emit)
        layout.addWidget(self.calib_btn)

        self.start_btn = QPushButton("开始")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(self.stop_btn)

        layout.addStretch()

        self.weight_btn = QPushButton("权重目录")
        self.weight_btn.setToolTip("选择自定义ONNX权重文件夹（按SN匹配）")
        self.weight_btn.clicked.connect(self._select_weight_dir)
        layout.addWidget(self.weight_btn)

        self.weight_label = QLabel("默认")
        self.weight_label.setStyleSheet("color: #888;")
        layout.addWidget(self.weight_label)

        layout.addStretch()

    def _start_scan(self):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            return
        self._scan_worker = DeviceScanWorker()
        self._scan_worker.devices_found.connect(self._on_scan_result)
        self._scan_worker.scan_error.connect(self._on_scan_error)
        self._scan_worker.finished.connect(lambda: setattr(self, '_scan_worker', None))
        self._scan_worker.start()

    def _on_scan_result(self, sns):
        # 保存当前选中的 SN，避免刷新时丢失用户选择
        previous_sn = self.sn_combo.currentText()
        self.sn_combo.clear()
        if sns:
            self.sn_combo.addItems(sns)
            # 恢复之前选中的 SN（如果仍在设备列表中）
            if previous_sn and previous_sn not in ("未找到设备", "错误"):
                idx = self.sn_combo.findText(previous_sn)
                if idx >= 0:
                    self.sn_combo.setCurrentIndex(idx)
            self._had_devices = True
        else:
            self.sn_combo.addItem("未找到设备")
            if self._had_devices and self._connected and not self._streaming:
                self._on_disconnect()
            self._had_devices = False
        self._update_buttons()

    def _on_scan_error(self, msg):
        self.sn_combo.clear()
        self.sn_combo.addItem("错误")
        self._update_buttons()

    def _on_connect(self):
        sn = self.sn_combo.currentText()
        if not sn or sn in ("未找到设备", "错误"):
            return
        try:
            finder = VTSDeviceFinder()
            config = finder.get_device_by_sn(sn)
            if config is None:
                QMessageBox.warning(self, "连接错误", f"无法获取设备信息，请确认传感器已连接 (SN: {sn})")
                return
            self.model_label.setText(f"型号: {config.name}")
            self.connect_request.emit(config)
            self._connected = True
            self._update_buttons()
        except VTSError as e:
            QMessageBox.warning(self, "连接错误", f"连接失败: {e}\n{e.suggestion}")

    def _on_disconnect(self):
        self.disconnect_request.emit()
        self._connected = False
        self._streaming = False
        self.model_label.setText("型号: --")
        self._refresh_timer.start(2000)
        self._update_buttons()

    def _on_start(self):
        self._streaming = True
        self._refresh_timer.stop()
        self._update_buttons()
        self.start_request.emit()

    def _on_stop(self):
        self._streaming = False
        self._refresh_timer.start(2000)
        self._update_buttons()
        self.stop_request.emit()

    def on_worker_connected(self):
        self._connected = True
        self._update_buttons()

    def on_worker_disconnected(self):
        self._connected = False
        self._streaming = False
        self._refresh_timer.start(2000)
        self._update_buttons()

    def _update_buttons(self):
        has_device = self.sn_combo.count() > 0 and self.sn_combo.currentText() not in ("未找到设备", "错误")
        self.connect_btn.setEnabled(has_device and not self._connected)
        self.refresh_btn.setEnabled(not self._connected)
        self.sn_combo.setEnabled(not self._connected)
        self.disconnect_btn.setEnabled(self._connected and not self._streaming)
        self.calib_btn.setEnabled(self._connected)
        self.start_btn.setEnabled(self._connected and not self._streaming)
        self.stop_btn.setEnabled(self._connected and self._streaming)
        self.marker_spin.setEnabled(not self._connected)
        self.weight_btn.setEnabled(not self._connected)

    def _restore_settings(self):
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                wd = data.get("weight_dir", "")
                if wd and os.path.isdir(wd):
                    self._weight_dir = wd
                    name = os.path.basename(wd) or wd
                    self.weight_label.setText(name)
                    self.weight_label.setStyleSheet("color: #2ecc71;")
        except Exception:
            pass

    def _save_settings(self):
        try:
            data = {"weight_dir": self._weight_dir or ""}
            os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
            with open(SETTINGS_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _select_weight_dir(self):
        start = self._weight_dir if self._weight_dir else ""
        d = QFileDialog.getExistingDirectory(self, "选择自定义权重目录", start)
        if d:
            self._weight_dir = d
            name = os.path.basename(d) or d
            self.weight_label.setText(name)
            self.weight_label.setStyleSheet("color: #2ecc71;")
            self._save_settings()

    def get_weight_dir(self):
        return self._weight_dir

    def get_marker_size(self):
        return self.marker_spin.value()

    def get_marker_offsets(self):
        return [10, 10, 10, 10]
