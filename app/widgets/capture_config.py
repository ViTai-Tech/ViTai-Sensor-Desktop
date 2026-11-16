# coding=utf-8
"""数据采集「配置项」组件（独立 widget，供主界面侧栏复用）。

只负责「配置 + 录制控制」：接收 DevicePanel 检测到的 SN → 为手指选 SN
（GF515）→ 开始采集 → 用「开始保存/停止保存」录制 episode。不包含实时
预览画面（预览由 MultiSensorViewer 负责），录制线程推来的帧通过 frame_ready
信号转发，供需要预览的宿主自行订阅。设备检测交由 DevicePanel 的刷新按钮。

为规避与 app.capture_client.vitai_capture_gui 的循环依赖，运行所需的
Worker / 手指标定组件 / 文案等都在 __init__ 中延迟导入。
"""
import os

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QGroupBox,
    QFormLayout,
    QScrollArea,
    QMessageBox,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal, Qt

# GF515 系列传感器才需要「左右手 / 手指」标定与配置导入；其它型号（如 GF225）
# 直接以 SN 录制。型号以 SN 前缀判定（与 multi_sensor_viewer / image_viewer 一致）。
GF515_PREFIX = "GF515"


def _is_gf515_fleet(sns):
    """已连接传感器是否全部为 GF515（空列表视为否）。"""
    return bool(sns) and all(s.upper().startswith(GF515_PREFIX) for s in sns)


class CaptureConfigWidget(QWidget):
    """实时预览之外的全部数据采集配置与录制控制。

    布局：标题行（标题 + 收起 + 语言切换）+ 按钮行 + 手指标定 +
    存储设置 + 保存内容 + 任务录制 + 底部状态行，整体可滚动。
    """

    # 录制线程推来的帧包 {"tiles", "stats", "labels"}（供需要预览的宿主订阅）
    frame_ready = pyqtSignal(object)
    # 出流状态变化（bool）
    streaming_ready = pyqtSignal(bool)
    # episode 状态变化 (bool on, str desc)
    episode_ready = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CaptureConfig")

        # 延迟导入，避免与 vitai_capture_gui 形成循环依赖
        from app.capture_client import vitai_capture_gui as cap
        self._cap = cap

        self._worker = None
        self._starting = False   # 启动流程进行中（含基线确认弹窗）→ 防重复触发
        self._sns = []  # 由 DevicePanel 注入的已扫描 SN（用于 UI / 手指标定）
        # 出流校验以渲染线程（Host 的 SensorWorker）为准，而不是扫描列表：
        # 只有真正在渲染出流的传感器才参与采集。为 None 表示宿主未接管，
        # 此时退回扫描列表，保证本组件脱离宿主仍可独立使用。
        self._rendering_sns_cb = None
        self._rendering_streaming = None
        # 渲染线程 worker 表提供者：采集复用同一台设备，避免二次开相机
        self._sensor_workers_cb = None
        self._gf515 = False  # 当前设备是否均为 GF515（决定是否显示手指标定）
        self._finger_widget = cap.FingerAssignWidget()

        self._setup_ui()
        self._refresh_path()
        self.setMinimumWidth(360)
        cap._RETRANSLATE.append(self.retranslate)

    def _setup_ui(self):
        cap = self._cap
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 内容区（可滚动，窗口矮时仍能操作全部配置）
        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        # 标题行：标题
        self._title = QLabel(cap.tr("app_title"))
        self._title.setObjectName("panelTitle")
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_row.addWidget(self._title)
        title_row.addStretch(1)
        v.addLayout(title_row)

        self._sub = QLabel(cap.tr("panel_subtitle"))
        self._sub.setObjectName("panelSub")
        self._sub.setWordWrap(True)
        v.addWidget(self._sub)
        v.addSpacing(4)

        # 采集/停止按钮（检测交由 DevicePanel 的刷新按钮；导入配置仅 GF515 显示）
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.import_btn = QPushButton(cap.tr("btn_import"))
        self.import_btn.setVisible(False)  # GF515 才显示，由 set_devices 控制
        self.start_btn = QPushButton(cap.tr("btn_start"))
        self.stop_btn = QPushButton(cap.tr("btn_stop"))
        self.recalib_btn = QPushButton(cap.tr("btn_recalib"))
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.recalib_btn.setEnabled(False)   # 出流后才可重新标定
        self.import_btn.clicked.connect(self._import_config)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._shutdown)
        self.recalib_btn.clicked.connect(self._recalibrate)
        btn_row.addWidget(self.import_btn)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.recalib_btn)
        v.addLayout(btn_row)

        v.addWidget(self._finger_widget)
        self._finger_widget.setVisible(False)  # GF515 才显示，由 set_devices 控制

        # 存储设置
        self._store = QGroupBox(cap.tr("store_title"))
        self._store_form = QFormLayout(self._store)
        self._store_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.task_edit = QLineEdit()
        self.task_edit.setPlaceholderText(cap.tr("task_placeholder"))
        self.task_edit.textChanged.connect(self._refresh_path)
        self._store_form.addRow(cap.tr("label_folder"), self.task_edit)
        self.path_label = QLabel("")
        self.path_label.setObjectName("panelSub")
        self.path_label.setWordWrap(True)
        self._store_form.addRow(cap.tr("label_path"), self.path_label)
        v.addWidget(self._store)

        # 保存内容：每类数据一个复选框；采集线程按勾选向 SDK 请求数据类型
        self._save_box = QGroupBox(cap.tr("save_group"))
        save_col = QVBoxLayout(self._save_box)
        self._save_checks = {}
        for key in cap.SAVE_TYPES:
            cb = cap.ColorCheckBox(cap.tr(f"save_{key}"))
            cb.setChecked(key not in ("diff", "raw"))   # 默认不勾 diff 视频与原生视频
            cb.toggled.connect(self._push_spec)
            self._save_checks[key] = cb
            save_col.addWidget(cb)
        v.addWidget(self._save_box)

        # 任务录制（episode）
        self._seg = QGroupBox(cap.tr("seg_title"))
        seg_form = QFormLayout(self._seg)
        seg_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        seg_btn_row = QHBoxLayout()
        seg_btn_row.setSpacing(8)
        self.seg_start_btn = QPushButton(cap.tr("btn_seg_start"))
        self.seg_end_btn = QPushButton(cap.tr("btn_seg_end"))
        self.seg_start_btn.setEnabled(False)
        self.seg_end_btn.setEnabled(False)
        self.seg_start_btn.clicked.connect(self._seg_start)
        self.seg_end_btn.clicked.connect(self._seg_stop)
        seg_btn_row.addWidget(self.seg_start_btn)
        seg_btn_row.addWidget(self.seg_end_btn)
        seg_btn_row.addStretch(1)
        seg_form.addRow("", seg_btn_row)
        self.seg_hint = QLabel(cap.tr("seg_hint_idle"))
        self._seg_hint_key = "seg_hint_idle"
        self.seg_hint.setObjectName("panelSub")
        self.seg_hint.setWordWrap(True)
        seg_form.addRow("", self.seg_hint)
        v.addWidget(self._seg)

        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll, 1)

        # 底部状态行（标定/出流进度、错误提示）：始终可见
        self.status = QLabel(cap.tr("status_ready"))
        self.status.setObjectName("panelSub")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

    def _refresh_path(self):
        task = self._cap._sanitize_name(self.task_edit.text(), "unnamed")
        self.path_label.setText(os.path.join(self._cap.DATA_DIR, task) + "/")

    # -- 文案刷新（语言切换按钮已移除，保留以备 UI 热更新复用）--
    def retranslate(self):
        """语言切换时刷新本组件所有静态文案。"""
        cap = self._cap
        self._title.setText(cap.tr("app_title"))
        self._sub.setText(cap.tr("panel_subtitle"))
        self.import_btn.setText(cap.tr("btn_import"))
        self.start_btn.setText(cap.tr("btn_start"))
        self.stop_btn.setText(cap.tr("btn_stop"))
        self._store.setTitle(cap.tr("store_title"))
        self.task_edit.setPlaceholderText(cap.tr("task_placeholder"))
        self._store_form.labelForField(self.task_edit).setText(cap.tr("label_folder"))
        self._store_form.labelForField(self.path_label).setText(cap.tr("label_path"))
        self._save_box.setTitle(cap.tr("save_group"))
        for key, cb in self._save_checks.items():
            cb.setText(cap.tr(f"save_{key}"))
        self.recalib_btn.setText(cap.tr("btn_recalib"))
        self._seg.setTitle(cap.tr("seg_title"))
        self.seg_start_btn.setText(cap.tr("btn_seg_start"))
        self.seg_end_btn.setText(cap.tr("btn_seg_end"))
        self.seg_hint.setText(cap.tr(self._seg_hint_key))
        if self._worker is None:
            self.status.setText(cap.tr("status_ready"))

    # -- 设备来源（来自 DevicePanel 的扫描结果）--
    def set_devices(self, sns):
        """注入扫描到的 SN，并按型号决定「左右手指标定 / 导入配置」的显隐。

        - 更新手指标定下拉框的可选项；
        - 仅当全部传感器均为 GF515 时，才显示左右手指标定与导入配置按钮
          （其它型号如 GF225 不需要手指映射，直接以 SN 录制）；
        - 这里只是扫描列表，真正的出流/采集 SN 以宿主渲染线程为准。
        """
        sns = list(sns or [])
        self._sns = sns
        self._finger_widget.set_sns(sns)
        gf515 = _is_gf515_fleet(sns)
        self._gf515 = gf515
        self._finger_widget.setVisible(gf515)
        self.import_btn.setVisible(gf515)
        self._sync_start_button()

    # -- 渲染线程（出流）状态：由宿主 ViewerPage 注入 --
    def set_rendering_sns_provider(self, cb):
        """注入回调，返回渲染线程当前真正在出流的 SN 列表。

        出流校验以渲染线程为准：扫描到但未连接/未出流的 SN 不应被采集。
        """
        self._rendering_sns_cb = cb
        self._sync_start_button()

    def set_rendering_streaming(self, on):
        """宿主通知渲染线程出流状态变化（SensorWorker 启动/停止）。"""
        self._rendering_streaming = bool(on)
        self._sync_start_button()
        self._refresh_recording_ui()

    def set_sensor_workers_provider(self, cb):
        """注入渲染线程 worker 表提供者。

        采集线程复用渲染线程已打开的设备（同一台相机不能并发打开）：
        由宿主返回 {sn: SensorWorker}；未提供时采集线程自行打开设备。
        """
        self._sensor_workers_cb = cb

    def _sensor_workers(self):
        """渲染线程 worker 表（None 表示独立打开设备）。"""
        if self._sensor_workers_cb is None:
            return None
        try:
            return self._sensor_workers_cb() or None
        except Exception:
            return None

    def _streaming_sns(self):
        """当前渲染线程真正在出流的 SN 列表。"""
        if self._rendering_sns_cb is None:
            return list(self._sns)
        try:
            return list(self._rendering_sns_cb() or [])
        except Exception:
            return []

    def _sync_start_button(self):
        """「开始采集」可用态：以渲染线程出流为准，未接管时退回扫描列表。"""
        rendering = self._rendering_streaming
        ready = bool(self._sns) if rendering is None else (
            rendering and bool(self._streaming_sns()))
        self.start_btn.setEnabled(ready and not self.is_running())

    def _streaming_active(self):
        """采集线程在跑且渲染线程在出流（宿主未接管时只看采集线程）。"""
        return (self._worker is not None and self._worker.isRunning()
                and (self._rendering_streaming is None
                     or self._rendering_streaming))

    def _import_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self._cap.tr("import_dialog_title"),
            self._cap.CONFIG_DIR, self._cap.tr("import_filter"))
        if not path:
            return
        ok, msg = self._finger_widget.import_config(path)
        if ok:
            self.status.setText(msg)
        else:
            QMessageBox.warning(self, self._cap.tr("import_warn_title"), msg)

    def _start(self):
        # 防重入：整个启动流程（含下面的基线确认弹窗）期间只允许一次。
        if self._starting or (self._worker is not None and self._worker.isRunning()):
            return
        self._starting = True
        try:
            # 出流校验以渲染线程为准：只采集真正在渲染出流的传感器，
            # 扫描到但未连接（未出流）的 SN 不参与采集。
            sensors_sn = self._streaming_sns()
            if not sensors_sn:
                QMessageBox.warning(self, self._cap.tr("start_no_device_title"),
                                    self._cap.tr("start_no_device_msg"))
                return
            if self._gf515:
                labels = self._finger_widget.labels_for(sensors_sn)
            else:
                labels = list(sensors_sn)
            # 基线保护：标定与首帧基线要求传感器处于无按压状态，开始前显式确认
            ans = QMessageBox.question(self, self._cap.tr("baseline_confirm_title"),
                                       self._cap.tr("baseline_confirm_msg"))
            if ans != QMessageBox.StandardButton.Yes:
                return
            if self._gf515:
                self._finger_widget.save_config()
            # 复用渲染线程已打开的设备（同一台相机不能并发打开）
            self._worker = self._cap.DataCaptureWorker(
                sensors_sn, labels, sensor_workers=self._sensor_workers())
            self._worker.status.connect(self._on_status)
            self._worker.streaming.connect(self._on_streaming)
            self._worker.streaming.connect(self.streaming_ready.emit)
            self._worker.episode_changed.connect(self._on_episode)
            self._worker.episode_changed.connect(self.episode_ready.emit)
            self._worker.frame.connect(self._on_frame)
            self._worker.finished.connect(self._on_finished)
            self._push_spec()  # 注入当前渲染模式，出流第一帧就按正确模式跑
            self._worker.start()
            self.start_btn.setEnabled(False)
            if self._gf515:
                self._finger_widget.set_enabled(False)
            self.stop_btn.setEnabled(True)
            self.status.setText(self._cap.tr("starting"))
        finally:
            self._starting = False

    def _shutdown(self):
        if self._worker is not None:
            self._worker.request_stop()
        self.setFocus()

    def is_running(self):
        """采集是否正在进行（数据线程在运行 = 正在出流/采集）。"""
        return self._worker is not None and self._worker.isRunning()

    def has_active_episode(self):
        """是否正在录制 episode（落盘中）。"""
        return (self._worker is not None and self._worker.isRunning()
                and self._worker.episode_active())

    def discard_recording(self):
        """请求丢弃当前正在录制的 episode（停止采集时直接删除，不落盘）。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.discard_episode()

    def shutdown(self):
        """关闭时调用：停采集 worker 线程并等待其释放传感器。"""
        if self._worker is not None:
            self._worker.request_stop()
            if self._worker.isRunning():
                if not self._worker.wait(5000):
                    self.status.setText(self._cap.tr("shutdown_timeout"))
        self._worker = None
        self._sync_start_button()

    # -- episode 录制动作 --
    def _seg_start(self):
        if not self._streaming_active():
            return
        task = self._cap._sanitize_name(self.task_edit.text(), "")
        if not task:
            QMessageBox.warning(self, self._cap.tr("seg_no_task_title"),
                                self._cap.tr("seg_no_task_msg"))
            return
        if not any(cb.isChecked() for cb in self._save_checks.values()):
            QMessageBox.warning(self, self._cap.tr("start_no_type_title"),
                                self._cap.tr("start_no_type_msg"))
            return
        self._push_spec()
        self._worker.start_episode()
        self.setFocus()

    def _seg_stop(self):
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.stop_episode()
        self.setFocus()

    def _push_spec(self):
        """把当前 UI 的录制选项（任务名 + 勾选的保存内容）推给采集线程。"""
        if self._worker is None:
            return
        save = {k for k, cb in self._save_checks.items() if cb.isChecked()}
        self._worker.set_spec({
            "task": self._cap._sanitize_name(self.task_edit.text(), ""),
            "save": save,
        })

    def _refresh_recording_ui(self):
        # 出流状态以渲染线程为准；宿主未提供渲染状态时退回采集线程自身判断。
        worker_active = self._worker is not None and self._worker.isRunning()
        streaming = self._streaming_active()
        # 正在录制时必须始终能停止，渲染线程中途停掉也不禁用「停止保存」。
        recording = worker_active and self._worker.episode_active()
        self.seg_start_btn.setEnabled(streaming and not recording)
        self.seg_end_btn.setEnabled(recording)
        if not streaming:
            self._seg_hint_key = "hint_no_stream"
        elif not recording:
            self._seg_hint_key = "hint_ready"
        else:
            self._seg_hint_key = "hint_recording"
        self.seg_hint.setText(self._cap.tr(self._seg_hint_key))

    # -- 信号槽 --
    def _on_status(self, msg):
        self.status.setText(msg)

    def _on_streaming(self, on):
        self.recalib_btn.setEnabled(bool(on))
        self._refresh_recording_ui()

    def _on_episode(self, on, desc):
        if on:
            self.status.setText(self._cap.tr("episode_recording", desc=desc))
        self._refresh_recording_ui()

    def _on_frame(self, payload):
        """采集线程推来的预览包：仅转发给需要预览的宿主，本组件不渲染。"""
        self.frame_ready.emit(payload)

    def _recalibrate(self):
        """「重新标定」：SDK 重设背景，深度/marker 基线随下一帧重建。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_recalibrate()
        self.setFocus()

    def _on_finished(self):
        if self._worker is not None:
            self._worker.wait(5000)
            self._worker = None
        self._sync_start_button()
        self._finger_widget.set_enabled(True)
        self.seg_start_btn.setEnabled(False)
        self.seg_end_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
