# coding=utf-8
"""按传感器类别划分的查看器页面。

每一个 ViewerPage 对应一个传感器类别（如「灵巧手」或「夹爪」），内部持有
自己的 DevicePanel（已按类别过滤设备列表）、MultiSensorViewer 与
CaptureConfigWidget，并独立管理该类别的采集 worker 与数据采集客户端。

这样多个类别可各自保持连接与采集状态，互不干扰，类似于 Web 端把
hand / gripper 拆成两个独立路由的查看器。主窗口 MainWindow 作为导航外壳，
负责在 Home 页与各 ViewerPage 之间切换。
"""

import time

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPushButton,
    QLabel,
    QMessageBox,
)

from app.sensor_worker import SensorWorker
from app.sensor_categories import get_category
from pyvitaisdk import VTSDataType

from app.widgets.device_panel import DevicePanel
from app.widgets.multi_sensor_viewer import MultiSensorViewer
from app.widgets.capture_config import CaptureConfigWidget


class ViewerPage(QWidget):
    back_requested = pyqtSignal()
    status_changed = pyqtSignal(str)
    timestamp_changed = pyqtSignal(str)
    transient_message = pyqtSignal(str, int)

    def __init__(self, cat_id, parent=None):
        super().__init__(parent)
        self.setObjectName("ViewerPage")
        self.category = cat_id

        self._workers = {}   # sn -> SensorWorker
        self._configs = []   # 已连接的传感器配置
        self._fps = {}       # sn -> 最近帧率
        # 停止超时、仍在自行退出的 worker（sn -> SensorWorker）。卡在 SDK 打开/
        # 标定里的线程不能强杀，只能等它自然退出后再回收。
        self._orphan_workers = {}
        self._pending_sns = set()  # 孤儿 worker 的 SN：相机可能仍被占用
        self._capture_visible = False  # 数据采集配置面板是否展开

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        # 局部导入，便于 UI 热更新后拿到 reload 过的类
        from app.widgets.device_panel import DevicePanel
        from app.widgets.multi_sensor_viewer import MultiSensorViewer
        from app.widgets.capture_config import CaptureConfigWidget

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.device_panel = DevicePanel()
        self.device_panel.set_category(self.category)
        cat = get_category(self.category)
        self.device_panel.set_title(cat["title"] if cat else "")
        root.addWidget(self.device_panel)

        # body 区用横向 QSplitter：左=查看器（弹性），右=采集配置
        self.body_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.body_splitter.setContentsMargins(0, 0, 0, 0)
        self.body_splitter.setHandleWidth(6)
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.setStretchFactor(0, 1)  # 查看器弹性占满
        self.body_splitter.setStretchFactor(1, 0)  # 配置项保持自身宽度
        root.addWidget(self.body_splitter, 1)

        self.multi_viewer = MultiSensorViewer()
        self.body_splitter.addWidget(self.multi_viewer)

        self.capture_panel = CaptureConfigWidget()
        # 出流校验以渲染线程为准：把当前 SensorWorker 的出流 SN 注入采集面板
        self.capture_panel.set_rendering_sns_provider(self._rendering_sns)
        # 采集复用渲染线程已打开的设备（同一台相机不能并发打开）
        self.capture_panel.set_sensor_workers_provider(
            self._sensor_workers_for_capture)
        self.capture_panel.setVisible(False)
        self.body_splitter.addWidget(self.capture_panel)

    # ------------------------------------------------------------------ #
    # 数据采集面板的显示 / 隐藏
    # ------------------------------------------------------------------ #
    def toggle_capture(self):
        if self.capture_panel.isVisible():
            self._hide_capture()
        else:
            self._show_capture()

    def _show_capture(self):
        # 采集面板只负责配置与录制，不干预 SensorWorker：实时渲染持续运行。
        self.capture_panel.setVisible(True)
        total = self.body_splitter.width()
        if total <= 0:
            total = self.width()
        cfg_w = 420
        viewer_w = max(360, total - cfg_w)
        self.body_splitter.setSizes([viewer_w, cfg_w])
        self.device_panel.set_capture_active(True)
        self._capture_visible = True

    def _hide_capture(self):
        if not self.confirm_leave_capture():
            return

        try:
            self.capture_panel.shutdown()
        except Exception:
            pass

        self.capture_panel.setVisible(False)
        self.device_panel.set_capture_active(False)
        self._capture_visible = False
        # 收起面板不影响渲染线程，无需重启传感器。

    def is_capturing(self):
        """数据采集是否正在进行。"""
        panel = getattr(self, "capture_panel", None)
        return panel is not None and panel.is_running()

    def confirm_leave_capture(self):
        """采集进行中时询问是否停止采集。

        「退出数据采集」与「返回首页」共用这段询问流程。
        返回 True 表示可以继续（未在采集，或用户确认停止）；False 表示用户取消。
        """
        if not self.is_capturing():
            return True

        ans = QMessageBox.question(
            self,
            "退出数据采集",
            "退出数据采集需要先停止当前的数据采集。\n"
            "确认要停止采集并退出吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return False

        if self.capture_panel.has_active_episode():
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle("保存录制")
            box.setText(
                "当前正在录制 episode，退出数据采集将停止采集。\n"
                "是否保存正在录制的 episode？"
            )
            save_btn = box.addButton(
                "保存并退出", QMessageBox.ButtonRole.YesRole)
            discard_btn = box.addButton(
                "丢弃并退出", QMessageBox.ButtonRole.NoRole)
            box.setDefaultButton(save_btn)
            box.exec()
            if box.clickedButton() is discard_btn:
                self.capture_panel.discard_recording()
        return True

    # ------------------------------------------------------------------ #
    # 信号连接
    # ------------------------------------------------------------------ #
    def _connect_signals(self):
        self.device_panel.connect_request.connect(self._on_connect)
        self.device_panel.disconnect_request.connect(self._on_disconnect)
        self.device_panel.start_request.connect(self._on_start)
        self.device_panel.stop_request.connect(self._on_stop)
        self.device_panel.calibrate_request.connect(self._on_calibrate)
        self.device_panel.data_capture_request.connect(self.toggle_capture)
        self.device_panel.devices_scanned.connect(self.capture_panel.set_devices)
        self.device_panel.back_requested.connect(self.back_requested.emit)

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
            panel.data_capture_request,
            panel.devices_scanned,
            panel.back_requested,
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
        if hasattr(self, "device_panel"):
            panel_state = self.device_panel.export_ui_state()
            self.device_panel.prepare_destroy()
            self._disconnect_panel_signals()

        if getattr(self, "capture_panel", None) is not None:
            try:
                self.capture_panel.shutdown()
            except Exception:
                pass

        if hasattr(self, "multi_viewer"):
            viewer_sns = self.multi_viewer.sensor_sns()

        reload_widget_modules()
        self._setup_ui()
        self._connect_signals()
        # 新面板需重新同步当前渲染线程出流状态
        self.capture_panel.set_rendering_streaming(bool(self._rendering_sns()))

        if panel_state is not None:
            self.device_panel.apply_ui_state(panel_state)
            # 重建后重新应用类别过滤
            self.device_panel.set_category(self.category)

        if viewer_sns:
            self.multi_viewer.set_sensors(viewer_sns)
            if self._workers:
                self.multi_viewer.set_loading(viewer_sns, False)

        if self.isVisible():
            self.device_panel.on_shown()

    # ------------------------------------------------------------------ #
    # worker 管理
    # ------------------------------------------------------------------ #
    def _sns(self):
        return [c.SN for c in self._configs]

    def _rendering_sns(self):
        """渲染线程（SensorWorker）当前正在出流的 SN 列表。"""
        return [sn for sn, worker in self._workers.items() if worker.isRunning()]

    def streaming_sns(self):
        """对外暴露正在出流（独占相机）的 SN 列表，供首页设备计数兜底。"""
        sns = self._rendering_sns()
        # 还在自行退出的孤儿 worker：相机可能尚未释放，扫描不到，也计入
        for sn in self._pending_sns:
            if sn not in sns:
                sns.append(sn)
        return sns

    def _sensor_workers_for_capture(self):
        """返回渲染线程 worker 表，供采集线程复用同一台设备（不再二次开相机）。"""
        return dict(self._workers)

    def _on_connect(self, configs):
        self._configs = list(configs)
        self._start_all_workers()

    def _start_all_workers(self):
        self._stop_all_workers()

        sns = self._sns()

        self.multi_viewer.set_sensors(sns)
        self.multi_viewer.set_loading(sns, True)

        self._fps.clear()

        marker_size = self.device_panel.get_marker_size()
        offsets = self.device_panel.get_marker_offsets()
        weight_dir = self.device_panel.get_weight_dir()

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
                lambda data, sn=sn: self._on_data_ready(sn, data)
            )
            worker.error_occurred.connect(
                lambda msg, sug, sn=sn: self._on_worker_error(sn, msg, sug)
            )
            worker.fps_updated.connect(
                lambda fps, sn=sn: self._on_fps_updated(sn, fps)
            )
            worker.weight_status.connect(self._on_weight_status)
            self._workers[sn] = worker
            worker.start()

        self.capture_panel.set_rendering_streaming(bool(self._rendering_sns()))
        self.set_status(f"采集中 ({len(self._configs)}个传感器)")

    def _on_disconnect(self):
        self._stop_all_workers()
        self._configs = []
        self._fps.clear()
        self.set_status("就绪")
        self.set_timestamp("时间戳: --")
        self.multi_viewer.reset()

    def _on_start(self):
        if not self._configs:
            return
        self._start_all_workers()

    def _on_stop(self):
        self._stop_all_workers()
        self._fps.clear()
        self.set_status("已停止")
        self.set_timestamp("时间戳: --")

    def _on_calibrate(self):
        for worker in self._workers.values():
            if worker.isRunning():
                worker.calibrate()
        self.transient_message.emit("已重新校准", 2000)

    def _on_data_ready(self, sn, data):
        self.multi_viewer.set_loading([sn], False)
        self.multi_viewer.update_data(sn, data)
        if VTSDataType.TIME_STAMP in data:
            self.set_timestamp(f"时间戳: {data[VTSDataType.TIME_STAMP]}")

    def _on_weight_status(self, applied, msg):
        self.transient_message.emit(msg, 3000)

    def _on_fps_updated(self, sn, fps):
        self._fps[sn] = fps
        self.multi_viewer.set_fps(sn, fps)

    def _on_worker_error(self, sn, message, suggestion):
        self._remove_worker(sn)
        self.capture_panel.set_rendering_streaming(bool(self._rendering_sns()))
        self.multi_viewer.remove_sensor(sn)
        if self._workers:
            self.transient_message.emit(f"[{sn}] {message}", 5000)
        else:
            full = message
            if suggestion:
                full += f"\n\n建议: {suggestion}"
            QMessageBox.critical(self, f"传感器错误 ({sn})", full)
            self.set_status(f"错误: {message[:60]}")
            self.device_panel.on_worker_disconnected()

    def _remove_worker(self, sn, timeout_ms=2000):
        """停止并移除某个传感器 worker。

        注意：不能用 terminate()。灵巧手有多个传感器，线程可能正卡在 SDK 的
        打开/标定或取帧调用里，terminate 会让 QThread 收不到 finished，
        随后的 wait() 就变成永久阻塞，界面表现为「点了没反应」。
        这里的做法是：超时后断开信号、转入孤儿列表，由定时器轮询等它自然退出。
        """
        worker = self._workers.pop(sn, None)
        if worker is None:
            return

        worker.stop()
        if not worker.isRunning() or worker.wait(timeout_ms):
            worker.deleteLater()
            return

        # 还在 SDK 调用里：断开信号（避免回调打到已重置的视图），延后回收
        worker.blockSignals(True)
        self._orphan_workers[sn] = worker
        self._pending_sns.add(sn)
        QTimer.singleShot(500, self._reap_orphan_workers)

    def _reap_orphan_workers(self):
        """轮询回收此前停止超时的 worker（线程退出后才能安全销毁）。"""
        for sn, worker in list(self._orphan_workers.items()):
            if worker.isRunning():
                continue
            worker.deleteLater()
            self._orphan_workers.pop(sn, None)
            self._pending_sns.discard(sn)
        if self._orphan_workers:
            QTimer.singleShot(1000, self._reap_orphan_workers)

    def _stop_all_workers(self):
        workers = dict(self._workers)
        self._workers.clear()

        # 先对全部 worker 发出停止信号，让它们并行收尾
        for worker in workers.values():
            worker.stop()

        # 共用同一个等待窗口（而不是逐个 wait 超时后累加）：
        # 灵巧手有多台传感器，串行等待会让返回首页卡上十几秒
        deadline = time.monotonic() + 2.0
        for worker in workers.values():
            if worker.isRunning():
                worker.wait(max(0, int((deadline - time.monotonic()) * 1000)))

        for sn, worker in workers.items():
            if worker.isRunning():
                # 仍卡在 SDK 调用里：断开信号，转入孤儿列表等它自己退出
                worker.blockSignals(True)
                self._orphan_workers[sn] = worker
                self._pending_sns.add(sn)
            else:
                worker.deleteLater()
        if self._orphan_workers:
            QTimer.singleShot(500, self._reap_orphan_workers)

        panel = getattr(self, "capture_panel", None)
        if panel is not None:
            panel.set_rendering_streaming(False)

    def release_devices(self):
        """返回首页时调用：断开连接、停采集与出流并释放相机。

        返回首页后不保留任何状态：采集面板收起、设备列表清空、查看器画面清空，
        下次进入该类别页面即全新开始（重新扫描 + 重新勾选 + 重新连接）。
        同时释放相机，保证首页能真实扫描到设备（SDK 枚举不到被占用的设备）。

        注意：采集进行中时的确认询问由 confirm_leave_capture() 负责，
        调用方（返回首页）需先调用它，用户取消则不应再调用本方法。
        """
        # 采集线程复用出流线程的帧数据，必须先停采集再停出流
        panel = getattr(self, "capture_panel", None)
        if panel is not None:
            try:
                panel.shutdown()
            except Exception:
                pass
            try:
                panel.set_devices([])
            except Exception:
                pass
            panel.setVisible(False)
            self._capture_visible = False
            self.device_panel.set_capture_active(False)

        # 停出流 + 清空连接配置 + 重置查看器画面
        self._on_disconnect()
        # 设备面板回到初始状态（清空下拉框与扫描缓存）
        self.device_panel.reset_state()

    def set_status(self, text):
        self.status_changed.emit(text)

    def set_timestamp(self, text):
        self.timestamp_changed.emit(text)

    def shutdown(self):
        """关闭时调用：停采集客户端与全部 worker。"""
        try:
            if getattr(self, "capture_panel", None) is not None:
                self.capture_panel.shutdown()
        except Exception:
            pass
        self._stop_all_workers()
        # 退出前给孤儿线程一点收尾时间，尽量避免 QThread 在运行中被析构
        for sn, worker in list(self._orphan_workers.items()):
            if worker.wait(1000):
                worker.deleteLater()
                self._orphan_workers.pop(sn, None)
                self._pending_sns.discard(sn)

    # ------------------------------------------------------------------ #
    # 可见性：页面隐藏时停止设备扫描（避免多页同时扫描），显示时恢复
    # ------------------------------------------------------------------ #
    def on_shown(self):
        """页面可见时刷新设备列表（主窗口切页后调用）。"""
        self.device_panel.on_shown()

    def showEvent(self, ev):
        super().showEvent(ev)
        self.device_panel.on_shown()

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self.device_panel.on_hidden()
