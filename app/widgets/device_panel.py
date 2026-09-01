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

from app.widgets.checkable_combo import CheckableComboBox


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
    connect_request = pyqtSignal(list)
    disconnect_request = pyqtSignal()
    start_request = pyqtSignal()
    stop_request = pyqtSignal()
    calibrate_request = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DevicePanel")
        self.setFixedHeight(80)
        self._connected = False
        self._streaming = False
        self._had_devices = False
        self._scan_worker = None
        self._weight_dir = None
        self._last_sns = None
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
        sn_layout.addWidget(QLabel("传感器:"))
        self.sn_check_combo = CheckableComboBox()
        self.sn_check_combo.setMinimumWidth(220)
        self.sn_check_combo.checked_changed.connect(self._update_buttons)
        sn_layout.addWidget(self.sn_check_combo)
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
        # 设备列表未变化时直接跳过，避免每 2 秒重建下拉框导致滚动位置被重置、
        # 勾选状态与展开状态被打断（下拉框"弹回顶部"）。
        sns_tuple = tuple(sorted(sns)) if sns else ()
        if self._last_sns is not None and sns_tuple == self._last_sns:
            self._update_buttons()
            return
        self._last_sns = sns_tuple

        # 保存当前勾选的 SN，避免刷新时丢失用户选择
        previous = set(self.sn_check_combo.checked_items())
        self.sn_check_combo.clear_items()
        if sns:
            self.sn_check_combo.add_select_all_item()
            for sn in sns:
                self.sn_check_combo.add_item(sn, checked=(sn in previous))
            self._had_devices = True
        else:
            self.sn_check_combo.add_placeholder("未找到设备")
            if self._had_devices and self._connected and not self._streaming:
                self._on_disconnect()
            self._had_devices = False
        self._update_buttons()

    def _on_scan_error(self, msg):
        # 出错后强制下一次成功扫描重建列表，避免把"错误"占位误判为未变化
        self._last_sns = None
        self.sn_check_combo.clear_items()
        self.sn_check_combo.add_placeholder("错误")
        self._update_buttons()

    def _on_connect(self):
        sns = self.sn_check_combo.checked_items()
        if not sns:
            QMessageBox.warning(
                self, "未选择传感器",
                "请先在下拉框中勾选至少一个传感器。"
            )
            return

        # 防呆：必须已选择权重目录
        if not self._weight_dir:
            QMessageBox.critical(
                self, "未选择权重目录",
                "请先点击【权重目录】按钮，选择包含传感器权重文件的目录后再连接。\n\n"
                "权重目录中应包含按传感器SN命名的子文件夹（如 S12345678/），"
                "每个子文件夹中存放对应的ONNX模型文件。"
            )
            return
        valid, err_msg = self._validate_weight_dir(self._weight_dir)
        if not valid:
            QMessageBox.critical(
                self, "权重目录无效",
                f"当前权重目录校验失败：{err_msg}\n\n"
                "请重新选择有效的权重目录后再连接。"
            )
            return

        configs = []
        finder = VTSDeviceFinder()
        for sn in sns:
            # 防呆：每个传感器都必须有对应的权重子文件夹
            if not self._find_sn_weight_folder(self._weight_dir, sn):
                QMessageBox.critical(
                    self, "未找到匹配的权重文件",
                    f"在权重目录中未找到与传感器SN({sn})匹配的子文件夹。\n\n"
                    f"请确认权重目录中存在名为\"{sn}\"或包含\"{sn}\"的子文件夹，"
                    "且其中包含ONNX模型文件。\n\n"
                    f"当前权重目录: {self._weight_dir}"
                )
                return
            config = finder.get_device_by_sn(sn)
            if config is None:
                QMessageBox.warning(
                    self, "连接错误",
                    f"无法获取设备信息，请确认传感器已连接 (SN: {sn})"
                )
                return
            configs.append(config)

        self.model_label.setText(f"型号: {len(configs)}个传感器")
        self.connect_request.emit(configs)
        self._connected = True
        self._streaming = True
        self._refresh_timer.stop()
        self._update_buttons()

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
        has_selection = len(self.sn_check_combo.checked_items()) > 0
        self.connect_btn.setEnabled(has_selection and not self._connected)
        self.refresh_btn.setEnabled(not self._connected)
        self.sn_check_combo.setEnabled(not self._connected)
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

    def _find_sn_weight_folder(self, weight_dir, sn):
        """在权重目录中查找匹配给定SN的子文件夹，返回文件夹路径或None."""
        if not weight_dir or not os.path.isdir(weight_dir) or not sn:
            return None
        sn_upper = sn.upper()
        for entry in os.listdir(weight_dir):
            entry_path = os.path.join(weight_dir, entry)
            if os.path.isdir(entry_path) and entry.upper() == sn_upper:
                return entry_path
        return None

    def _validate_weight_dir(self, d):
        """校验权重目录是否包含至少一个有效子文件夹（含模型文件且名称像合法SN）."""
        if not d or not os.path.isdir(d):
            return False, "目录不存在"
        has_valid = False
        for entry in os.listdir(d):
            entry_path = os.path.join(d, entry)
            if not os.path.isdir(entry_path):
                continue
            # 子文件夹必须含文件
            has_files = False
            for f in os.listdir(entry_path):
                if os.path.isfile(os.path.join(entry_path, f)):
                    has_files = True
                    break
            if not has_files:
                continue
            # 子文件夹名必须像合法SN（至少4位字母或数字组合）
            if len(entry) >= 4 and any(c.isdigit() for c in entry) and any(c.isalpha() for c in entry):
                has_valid = True
                break
        if not has_valid:
            return False, "所选目录中未找到任何符合命名规范的权重子文件夹\n（子文件夹名应为传感器SN，至少包含字母+数字，如 S12345678 或 GF515I）"
        return True, ""

    def _select_weight_dir(self):
        start = self._weight_dir if self._weight_dir else ""
        d = QFileDialog.getExistingDirectory(self, "选择自定义权重目录", start)
        if d:
            valid, err_msg = self._validate_weight_dir(d)
            if not valid:
                QMessageBox.critical(
                    self, "权重目录无效",
                    f"权重目录校验失败：{err_msg}\n\n"
                    "请选择一个包含按传感器SN命名的子文件夹的目录，"
                    "每个子文件夹中应包含对应的ONNX权重文件。"
                )
                return
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
