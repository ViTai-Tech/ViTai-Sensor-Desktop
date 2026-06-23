import os
import shutil

from PyQt6.QtCore import QThread, pyqtSignal
from pyvitaisdk import VTSensor, VTSensorType, VTSDataType, VTSError


def _get_sdk_weights_dir():
    import pyvitaisdk
    base = os.path.dirname(pyvitaisdk.__file__)
    return os.path.join(base, "vts", "reconstruct3d", "weights")


def _apply_custom_weights(weight_dir, sn, sdk_weights):
    """Match custom weights by SN and copy into SDK weights directory."""   
    if not os.path.isdir(weight_dir):
        return False

    # Try to find folder matching this SN
    sn_folder = None
    sn_short = sn.upper()
    for entry in os.listdir(weight_dir):
        entry_path = os.path.join(weight_dir, entry)
        if os.path.isdir(entry_path) and entry.upper() in sn_short:
            sn_folder = entry_path
            break
    if not sn_folder:
        # Try partial match
        for entry in os.listdir(weight_dir):
            entry_path = os.path.join(weight_dir, entry)
            if os.path.isdir(entry_path) and sn_short in entry.upper():
                sn_folder = entry_path
                break
    if not sn_folder:
        return False

    applied = False
    for f in os.listdir(sn_folder):
        src = os.path.join(sn_folder, f)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(sdk_weights, f)
        shutil.copy2(src, dst)
        applied = True
    return applied


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
            weight_applied = False
            if self._weight_dir:
                sn = self._config.SN if hasattr(self._config, 'SN') else ''
                sdk_weights = _get_sdk_weights_dir()
                weight_applied = _apply_custom_weights(self._weight_dir, sn, sdk_weights)
                if weight_applied:
                    self.weight_status.emit(True, f"已应用自定义权重 ({sn})")
                else:
                    self.weight_status.emit(False, f"未找到SN匹配的权重，使用默认")

            self._vtsensor = VTSensor(
                config=self._config,
                marker_size=self._marker_size,
                marker_offsets=self._marker_offsets,
            )
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
