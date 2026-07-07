import os

from PyQt6.QtCore import QThread, pyqtSignal
from pyvitaisdk import VTSensor, VTSensorType, VTSDataType, VTSError


def _find_model_path(weight_dir, sn):
    """在权重目录中查找匹配SN的模型文件，返回 (model_path, depth_model_path) 或 (None, None).

    查找规则：
    1. weight_dir/{sn}/{sn}.onnx.enc 作为 force_model_path
    2. weight_dir/{sn}/ 下任意 .onnx.enc 文件作为 force_model_path
    3. weight_dir/{sn}/depth/ 下的 .onnx 文件作为 depth model
    """
    if not weight_dir or not os.path.isdir(weight_dir) or not sn:
        return None, None

    sn_upper = sn.upper()
    sn_folder = None
    for entry in os.listdir(weight_dir):
        entry_path = os.path.join(weight_dir, entry)
        if os.path.isdir(entry_path) and entry.upper() == sn_upper:
            sn_folder = entry_path
            break
    if not sn_folder:
        return None, None

    # 查找 force model: 优先 {sn}.onnx.enc，其次任意 .onnx.enc
    force_model = None
    preferred = os.path.join(sn_folder, f"{sn}.onnx.enc")
    if os.path.isfile(preferred):
        force_model = preferred
    else:
        for f in os.listdir(sn_folder):
            fpath = os.path.join(sn_folder, f)
            if os.path.isfile(fpath) and f.endswith(".onnx.enc"):
                force_model = fpath
                break

    return force_model, None


class SensorWorker(QThread):
    data_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str, str)
    sensor_info = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    weight_status = pyqtSignal(bool, str)

    def __init__(self, config, marker_size=9, marker_offsets=None, weight_dir=None,
                 display_freq=15, parent=None):
        super().__init__(parent)
        self._config = config
        self._marker_size = marker_size
        self._marker_offsets = marker_offsets or [10, 10, 10, 10]
        self._weight_dir = weight_dir
        self._display_interval = 1.0 / max(display_freq, 1)
        self._running = False
        self._vtsensor = None
        self._do_calibrate = False

    def run(self):
        try:
            force_model_path = None
            if self._weight_dir:
                sn = self._config.SN if hasattr(self._config, 'SN') else ''
                if not sn:
                    self.error_occurred.emit(
                        "无法获取传感器序列号(SN)",
                        "请检查传感器是否正确连接"
                    )
                    return
                force_model_path, _ = _find_model_path(self._weight_dir, sn)
                if force_model_path:
                    self.weight_status.emit(True, f"已加载自定义权重: {os.path.basename(force_model_path)}")
                else:
                    self.error_occurred.emit(
                        f"未找到与当前传感器SN({sn})匹配的权重文件",
                        f"请在权重目录中确认:\n"
                        f"1. 存在名为 \"{sn}\" 的子文件夹\n"
                        f"2. 子文件夹中包含 \"{sn}.onnx.enc\" 或其他 .onnx.enc 文件\n\n"
                        f"当前权重目录: {self._weight_dir}"
                    )
                    return
            else:
                self.error_occurred.emit(
                    "未设置权重目录",
                    "请先选择自定义权重目录"
                )
                return

            kwargs = dict(
                config=self._config,
                marker_size=self._marker_size,
                marker_offsets=self._marker_offsets,
            )
            if force_model_path:
                kwargs["force_model_path"] = force_model_path

            self._vtsensor = VTSensor(**kwargs)
            self.sensor_info.emit(str(self._vtsensor.sensor_type.value))
            self._vtsensor.calibrate()
            self.connected.emit()

            import time

            frame_count = 0
            last_fps_time = time.time()
            last_display_time = 0.0
            self._running = True

            while self._running:
                if self._do_calibrate:
                    self._vtsensor.calibrate()
                    self._do_calibrate = False

                data = self._vtsensor.collect_sensor_data(
                    VTSDataType.TIME_STAMP,
                    VTSDataType.RAW_IMG,
                    VTSDataType.WARPED_IMG,
                    VTSDataType.DIFF_IMG,
                    VTSDataType.DEPTH_MAP,
                    VTSDataType.MARKER_IMG,
                    VTSDataType.MARKER_ORIGIN_VECTOR,
                    VTSDataType.MARKER_CURRENT_VECTOR,
                    VTSDataType.MARKER_OFFSET_VECTOR,
                    VTSDataType.XYZ_VECTOR,
                    VTSDataType.FORCE6D_VECTOR,
                    VTSDataType.SLIP_STATE,
                )

                now = time.time()
                if now - last_display_time >= self._display_interval:
                    self.data_ready.emit(data)
                    last_display_time = now

                frame_count += 1
                elapsed = now - last_fps_time
                if elapsed >= 1.0:
                    self.fps_updated.emit(frame_count / elapsed)
                    frame_count = 0
                    last_fps_time = now

        except VTSError as e:
            self.error_occurred.emit(str(e), str(e.suggestion))
        except Exception as e:
            self.error_occurred.emit(str(e), "")
        finally:
            if self._vtsensor:
                self._vtsensor.release()
            self._running = False
            self.disconnected.emit()

    def stop(self):
        self._running = False

    def calibrate(self):
        self._do_calibrate = True
