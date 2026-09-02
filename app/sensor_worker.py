import os
import threading

from PyQt6.QtCore import QThread, pyqtSignal
from pyvitaisdk import VTSensor, VTSensorType, VTSDataType, VTSError

try:
    # 帧暂时不可用（区别于相机断开）：采集线程被抢占 CPU 时队列会短暂为空，
    # 这种错误可重试，不应直接把整个传感器行移除。
    from pyvitaisdk.common.exceptions import FrameNotAvailableError
except Exception:  # 安装包结构不一致时退回 None，按致命错误处理
    FrameNotAvailableError = None


# 串行打开锁：多个 ViTai 传感器共享 USB2.0 带宽，必须「逐个」打开+标定，
# 不能 N 个线程同时开（前面已开、正在传图的传感器会抢带宽，导致后面标定极慢/失败）。
_OPEN_LOCK = threading.Lock()

# 轻量数据：每帧都采，仅从视频流取帧，不含任何推理，CPU 开销可忽略。
_LIGHT_DATA_TYPES = (
    VTSDataType.TIME_STAMP,
    VTSDataType.RAW_IMG,
    VTSDataType.WARPED_IMG,
)

# 重量数据：深度重建(ONNX+泊松DST) + marker 光流跟踪 + 六维力估计(MLP)。
# 这些是主要 CPU 开销来源。多个传感器并发做这些重推理时，会占满 CPU，
# 导致 MSMF 采集线程饿死（在 Windows 上表现为：只有先启动的那个传感器还能出帧，
# 其余传感器「无信号」、折线图不动）。SDK 的 collect_sensor_data 本身是线程安全的
# （每传感器各自持有独立的 ONNX session / 光流跟踪器），问题在 CPU 抢占而非共享状态。
#
# 因此：重量数据按独立频率限速（heavy_freq），轻量数据仍按显示频率全速更新，
# 既保留「并行显示」，又把总 CPU 压低到 Windows 能扛住的水平。
_HEAVY_DATA_TYPES = (
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
                 display_freq=15, heavy_freq=8, parent=None):
        super().__init__(parent)
        self._config = config
        self._marker_size = marker_size
        self._marker_offsets = marker_offsets or [10, 10, 10, 10]
        self._weight_dir = weight_dir
        self._display_interval = 1.0 / max(display_freq, 1)
        self._heavy_interval = 1.0 / max(heavy_freq, 1)
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

            with _OPEN_LOCK:
                # 打开+标定在锁内串行进行，避免 MSMF 并发开流竞态
                self._vtsensor = VTSensor(**kwargs)
                self._vtsensor.calibrate()
            self.sensor_info.emit(str(self._vtsensor.sensor_type.value))
            self.connected.emit()

            import time

            frame_count = 0
            last_fps_time = time.time()
            last_display_time = 0.0
            last_heavy_time = 0.0
            merged = {}          # 合并后的最新数据：轻量帧只刷新 warp/raw/time，重量结果保留上一帧
            consecutive_misses = 0
            self._running = True

            while self._running:
                if self._do_calibrate:
                    self._vtsensor.calibrate()
                    self._do_calibrate = False
                    merged = {}
                    last_heavy_time = 0.0
                    continue

                now = time.time()
                if now - last_heavy_time >= self._heavy_interval:
                    # 重量推理帧：深度 + marker + 力，按 heavy_freq 限速。
                    data_types = _HEAVY_DATA_TYPES + _LIGHT_DATA_TYPES
                    last_heavy_time = now
                else:
                    # 轻量帧：只取 warp/raw + 时间戳，重量结果沿用上一帧。
                    data_types = _LIGHT_DATA_TYPES

                try:
                    data = self._vtsensor.collect_sensor_data(*data_types)
                except VTSError as e:
                    if FrameNotAvailableError is not None and isinstance(e, FrameNotAvailableError):
                        # 采集线程被抢 CPU 导致队列短暂为空：跳过本帧，不整行移除。
                        # 连续失败过多时才向上抛，避免相机真的卡死时无限空转。
                        consecutive_misses += 1
                        if consecutive_misses >= 30:
                            raise
                        time.sleep(0.01)
                        continue
                    raise
                consecutive_misses = 0

                merged.update(data)

                if now - last_display_time >= self._display_interval:
                    self.data_ready.emit(dict(merged))
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
