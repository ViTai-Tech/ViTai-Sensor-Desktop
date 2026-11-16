#!/usr/bin/env python3
# coding=utf-8
"""
ViTai 传感器数据采集（简化版，仅保留实时预览 + 采集一个功能）。

与 pipeline_gui.py 的第一个标签页等价，去掉了侧边栏导航、其它流水线步骤、
COS 上传/下载与子进程运行逻辑，只留「ViTai 传感器实时预览 + 按「开始保存/停止保存」录制 episode」。

用法:
    python vitai_capture_gui.py                              # 启动界面
    ... --dark                                               # 深色主题
    ... --smoke                                              # 无显示器冒烟（配合 QT_QPA_PLATFORM=offscreen）

依赖: PyQt6、pyvitaisdk、opencv、numpy（安装步骤见 README）。
力模型解析 / 滑动状态等工具已内联在本文件内，无其它项目内依赖。
"""
import csv
import json
import os
import queue
import shutil
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QRect, QRectF
from PyQt6.QtGui import (
    QColor, QGuiApplication, QImage, QPainter, QPalette, QPen, QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton, QScrollArea,
    QSizePolicy, QGroupBox, QGridLayout, QMessageBox, QSplitter, QFileDialog,
)

# --------------------------------------------------------------------------- #
# 路径与常量
# --------------------------------------------------------------------------- #
DATA_AE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DATA_AE, "data")
CONFIG_DIR = os.path.join(DATA_AE, "config")     # 配置文件目录（本目录下的 config/）

# 手指 ↔ SN 标定：固定左右手两组，每手 5 指；下拉框全局唯一。
# 配置保存到本程序目录下 config/sensor_fingers.json（独立配置文件，不与其他程序共享；
# 路径相对本脚本计算，客户把 vitai_capture_client 部署到任意目录都能正常读写）。
HANDS = ["right", "left"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
CAPTURE_CONFIG_PATH = os.path.join(CONFIG_DIR, "sensor_fingers.json")


# --------------------------------------------------------------------------- #
# 国际化：中/英切换（LANG 全局 + tr() 取词 + set_language() 触发控件刷新）
# --------------------------------------------------------------------------- #
LANG = "zh"   # 当前语言：zh / en

STRINGS = {
    # 应用 / 面板
    "app_title": {"zh": "ViTai 传感器数据采集", "en": "ViTai Sensor Data Capture"},
    "panel_subtitle": {
        "zh": "检测设备 → 为手指选 SN → 开始采集 → 用「开始保存/停止保存」录制 episode。q / Esc 停止。",
        "en": "Detect → assign SN to fingers → start capture → record episodes with Start/Stop Saving. q / Esc to stop.",
    },
    # 手 / 手指
    "hand_right": {"zh": "右手", "en": "Right hand"},
    "hand_left": {"zh": "左手", "en": "Left hand"},
    "finger_thumb": {"zh": "拇指", "en": "Thumb"},
    "finger_index": {"zh": "食指", "en": "Index"},
    "finger_middle": {"zh": "中指", "en": "Middle"},
    "finger_ring": {"zh": "无名指", "en": "Ring"},
    "finger_pinky": {"zh": "小指", "en": "Pinky"},
    "fingers_count": {"zh": "{hand} · {n} 根手指", "en": "{hand} · {n} fingers"},
    "unassigned": {"zh": "（未分配）", "en": "(Unassigned)"},
    # 按钮
    "btn_detect": {"zh": "检测设备", "en": "Detect Devices"},
    "btn_import": {"zh": "导入配置 [可选]", "en": "Import Config [Optional]"},
    "btn_start": {"zh": "开始采集", "en": "Start Capture"},
    "btn_stop": {"zh": "停止采集", "en": "Stop Capture"},
    "btn_recalib": {"zh": "重新标定", "en": "Recalibrate"},
    "btn_back_viewer": {"zh": "返回查看器", "en": "Back to Viewer"},
    # 存储设置
    "store_title": {"zh": "存储设置", "en": "Storage"},
    "task_placeholder": {"zh": "例如 cloth；数据存到 data/<此名>/", "en": "e.g. cloth; saved to data/<name>/"},
    "label_folder": {"zh": "文件夹名", "en": "Folder name"},
    "label_path": {"zh": "保存路径", "en": "Save path"},
    # 保存内容（按数据类型勾选）
    "save_group": {"zh": "保存内容", "en": "Save Content"},
    "save_warped": {"zh": "矫正视频 warped（含背景图 bg.png）", "en": "Warped video (incl. bg.png)"},
    "save_diff": {"zh": "差异图 diff 视频", "en": "Diff image video"},
    "save_depth": {"zh": "深度图 depth_map（含基线 depth_baseline）", "en": "Depth map (incl. baseline)"},
    "save_marker": {"zh": "Marker 坐标（含基线 marker_baseline）", "en": "Marker positions (incl. baseline)"},
    "save_force": {"zh": "六维力 force", "en": "6-axis force"},
    "save_slip": {"zh": "滑动检测 slip", "en": "Slip detection"},
    "save_raw": {"zh": "原生视频 raw", "en": "Raw video"},
    # 任务录制
    "seg_title": {"zh": "任务录制", "en": "Task Recording"},
    "btn_seg_start": {"zh": "开始保存", "en": "Start Saving"},
    "btn_seg_end": {"zh": "停止保存", "en": "Stop Saving"},
    "seg_hint_idle": {
        "zh": "点「开始保存」开始录制一个 episode，点「停止保存」结束。",
        "en": "Click Start Saving to record an episode, Stop Saving to end.",
    },
    # 状态 / 预览
    "status_ready": {"zh": "就绪。", "en": "Ready."},
    "preview_no_stream": {"zh": "尚未出流", "en": "No stream"},
    # DetectWorker
    "detect_fail": {"zh": "发现设备失败: {e}（suggestion: {s}）", "en": "Detect failed: {e} (suggestion: {s})"},
    "detect_found": {"zh": "发现 {n} 台设备，请在下方为每根手指选择对应的 SN。", "en": "Found {n} device(s). Assign a SN to each finger below."},
    # DataCaptureWorker
    "cap_opening": {"zh": "开始打开并标定 {n} 台传感器（可能较慢，请耐心等待）...", "en": "Opening and calibrating {n} sensor(s) (may take a while)..."},
    "cap_not_found": {"zh": "未找到设备 {sn}（{lbl}），跳过。", "en": "Device {sn} ({lbl}) not found, skipped."},
    "cap_calibrating": {"zh": "标定 {sn}（{lbl}）...", "en": "Calibrating {sn} ({lbl})..."},
    "cap_opened": {"zh": "已打开并标定: {sn}", "en": "Opened and calibrated: {sn}"},
    "cap_open_fail": {"zh": "打开 {sn} 失败: {e}", "en": "Failed to open {sn}: {e}"},
    "cap_none": {"zh": "没有成功打开任何传感器。", "en": "No sensor opened successfully."},
    "cap_streaming": {"zh": "开始出流 {n} 路（点「开始保存」录制 episode，q / Esc 停止）", "en": "Streaming {n} channel(s) (click Start Saving to record, q / Esc to stop)"},
    "cap_read_err": {"zh": "读取 {sn} 出错: {e}", "en": "Error reading {sn}: {e}"},
    "cap_stopped": {"zh": "已停止采集。", "en": "Capture stopped."},
    "cap_fatal": {"zh": "采集线程异常终止: {e}", "en": "Capture thread crashed: {e}"},
    "cap_baseline": {"zh": "{sn} 首帧基线 min={min} max={max} mean={mean}", "en": "{sn} first-frame baseline min={min} max={max} mean={mean}"},
    "cap_force_model": {
        "zh": "{sn} 力模型: {name}（来源 {src}；标定期间该指请勿按压）",
        "en": "{sn} force model: {name} (from {src}; keep this finger unloaded while calibrating)",
    },
    "cap_force_fail": {
        "zh": "{sn} 力模型不可用: {e} —— 该传感器不记录力",
        "en": "{sn} force model unavailable: {e} - this sensor records no force",
    },
    "cap_slip_fail": {
        "zh": "{sn} 滑动检测不可用: {e} —— 该传感器不记录 slip",
        "en": "{sn} slip detection unavailable: {e} - this sensor records no slip",
    },
    "recalibrated": {"zh": "已重新标定，基线已重建。", "en": "Recalibrated; baselines rebuilt."},
    "episode_ended_detail": {
        "zh": "本段 episode 已结束：{frames} 帧，约 {size} MB，写盘丢帧 {dropped}。",
        "en": "Episode ended: {frames} frames, ~{size} MB, {dropped} writer-dropped.",
    },
    "episode_discarded": {
        "zh": "已丢弃当前录制的 episode：{desc}",
        "en": "Discarded current recording episode: {desc}",
    },
    "cap_episode_start": {"zh": "开始录制 episode -> {desc}", "en": "Started recording episode -> {desc}"},
    "start_no_type_title": {"zh": "未勾选保存内容", "en": "Nothing selected"},
    "start_no_type_msg": {
        "zh": "请至少勾选一项保存内容。",
        "en": "Select at least one data type to save.",
    },
    # 配置导入
    "cfg_read_fail": {"zh": "读取配置文件失败: {e}", "en": "Failed to read config file: {e}"},
    "cfg_format_err": {"zh": "配置文件格式错误：顶层应为 {\"right\": {...}, \"left\": {...}}。", "en": "Invalid config format: top level should be {\"right\": {...}, \"left\": {...}}."},
    "cfg_imported": {"zh": "已导入 {applied} 项，自动匹配 {matched} 项。", "en": "Imported {applied} item(s), matched {matched}."},
    "cfg_unmatched": {"zh": "（未匹配项：设备未检测到或 SN 不一致，检测到后会再自动匹配）", "en": " (unmatched: device not detected or SN mismatch; will auto-match once detected)"},
    # 采集面板动作
    "detecting": {"zh": "正在检测设备…", "en": "Detecting devices..."},
    "import_dialog_title": {"zh": "导入配置文件", "en": "Import config file"},
    "import_filter": {"zh": "配置文件 (*.json)", "en": "Config file (*.json)"},
    "import_warn_title": {"zh": "导入配置", "en": "Import config"},
    "start_no_finger_title": {"zh": "未分配手指", "en": "No finger assigned"},
    "start_no_finger_msg": {"zh": "请先检测设备，并为至少一根手指选择对应的 SN。", "en": "Detect devices and assign a SN to at least one finger first."},
    "start_dup_title": {"zh": "SN 重复", "en": "Duplicate SN"},
    "start_dup_msg": {"zh": "存在重复的 SN 分配，请检查下拉框。", "en": "Duplicate SN assignment; check the dropdowns."},
    "start_no_device_title": {"zh": "未检测到设备", "en": "No devices detected"},
    "start_no_device_msg": {"zh": "请先在顶部 DevicePanel 刷新并连接传感器。", "en": "Refresh and connect sensors in the DevicePanel first."},
    "starting": {"zh": "正在标定并出流…", "en": "Calibrating and streaming..."},
    "shutdown_timeout": {"zh": "等待采集线程退出超时。", "en": "Timed out waiting for capture thread to exit."},
    "seg_no_task_title": {"zh": "未填写文件夹名", "en": "No folder name"},
    "seg_no_task_msg": {"zh": "请先在「存储设置」里填写文件夹名。", "en": "Fill in a folder name under Storage first."},
    "baseline_confirm_title": {"zh": "提示", "en": "Info"},
    "baseline_confirm_msg": {
        "zh": "请确认所有传感器当前未被按压。是否继续？",
        "en": "Make sure no sensor is being pressed. Continue?",
    },
    "hint_no_stream": {"zh": "尚未出流——请先点「开始采集」。", "en": "Not streaming — click Start Capture first."},
    "hint_ready": {"zh": "点「开始保存」开始录制一个 episode。", "en": "Click Start Saving to record an episode."},
    "hint_recording": {"zh": "录制中：点「停止保存」结束本段 episode_<N>。", "en": "Recording: click Stop Saving to end episode_<N>."},
    "episode_recording": {"zh": "录制中 -> {desc}", "en": "Recording -> {desc}"},
    # 主窗口 / 菜单
    "menu_view": {"zh": "视图", "en": "View"},
    "menu_dark": {"zh": "深色模式", "en": "Dark Mode"},
    # 语言切换按钮（显示目标语言）
    "lang_switch": {"zh": "EN", "en": "中文"},
}


def tr(key, **kw):
    """返回当前语言下 key 的文案；kw 用作 str.format 占位。"""
    s = STRINGS[key][LANG]
    return s.format(**kw) if kw else s


_RETRANSLATE = []   # 语言切换回调，刷新静态控件文案


def set_language(lang):
    global LANG
    LANG = lang
    for cb in _RETRANSLATE:
        cb()


def _finger_labels():
    """按采集顺序生成 label 列表：右手拇指→小指，再左手拇指→小指。"""
    return [f"{hand}_{finger}" for hand in HANDS for finger in FINGERS]


# --------------------------------------------------------------------------- #
# 力模型解析 / 滑动状态（内联实现；原先复用 debug_read_all_sensors.py，现已独立）
# --------------------------------------------------------------------------- #
NOISE_FLOOR = 0.016   # 形变死区：深度差绝对值低于此值视为噪声（与预览面板上色用同一阈值）

# 滑动状态（VTSlipState）：1 CONTACT / 2 INCIPIENT_SLIP / 3 PARTIAL_SLIP
# 4 COMPLETE_SLIP / 5 NO_OBJ / 6 STEADY_HOLD；-1 表示本帧没取到
SLIP_STATE_DOC = (
    "1=CONTACT 2=INCIPIENT_SLIP 3=PARTIAL_SLIP 4=COMPLETE_SLIP 5=NO_OBJ "
    "6=STEADY_HOLD; -1 = not available"
)

# 厂商按 SN 下发的力权重目录（自动索引，无需额外配置）：
#     LM_sensor_force_weight/<SN>/<SN>.onnx.enc|.trt.enc|.rknn.enc + normalization_params.json
# 依次在 VITAI_FORCE_WEIGHT_DIR 环境变量、脚本目录、DATA_DIR、当前目录、家目录下查找同名目录。
FORCE_WEIGHT_DIRNAME = "LM_sensor_force_weight"
FORCE_WEIGHT_ENV = "VITAI_FORCE_WEIGHT_DIR"
# 后端优先级；每个后端先试加密版（.enc）再试明文。onnx 是通用后端（onnxruntime 是力模块的硬依赖），
# trt/rknn 只在对应设备上可用 —— 可用性交给 vitai_force 自己的运行时检查否决。
FORCE_WEIGHT_BACKENDS = (".onnx", ".trt", ".rknn")

def _force_weight_dirs() -> List[str]:
    """LM_sensor_force_weight 的候选目录（只保留真实存在的，去重保序）。"""
    cands = []
    env = (os.environ.get(FORCE_WEIGHT_ENV) or "").strip()
    if env:
        # 环境变量既可以直接指向权重目录，也可以指向它的父目录
        cands.append(env if os.path.basename(env.rstrip(os.sep)) == FORCE_WEIGHT_DIRNAME
                     else os.path.join(env, FORCE_WEIGHT_DIRNAME))
    for base in (DATA_AE, DATA_DIR, os.getcwd(), os.path.expanduser("~")):
        cands.append(os.path.join(base, FORCE_WEIGHT_DIRNAME))
    out, seen = [], set()
    for c in cands:
        c = os.path.abspath(c)
        if c not in seen and os.path.isdir(c):
            seen.add(c)
            out.append(c)
    return out


def _force_weight_usable(path: str) -> bool:
    """该权重文件在当前机器上能否加载（后缀/运行时）；用 vitai_force 自己的检查，避免自行猜测。"""
    try:
        from vitai_force.utils import resolve_force_model_path
    except Exception:  # noqa: BLE001 没有 vitai_force 时只按后缀兜底
        return path.lower().endswith((".onnx", ".onnx.enc"))
    try:
        resolve_force_model_path(path)
        return True
    except Exception:  # noqa: BLE001
        return False


_FORCE_WEIGHT_INDEX: Dict[str, Tuple[float, Dict[str, str]]] = {}


def _scan_force_weight_dir(root: str, force: bool = False) -> Dict[str, str]:
    """扫一个权重根目录 -> {SN: 权重文件路径}（带缓存，目录内容变了自动重扫）。

    每个 <SN>/ 目录里按 FORCE_WEIGHT_BACKENDS 的顺序挑第一个能在本机加载的权重；
    文件不以 SN 命名时退回到「目录里该后端唯一的那个文件」。force=True 忽略缓存重扫。
    """
    try:
        stamp = os.stat(root).st_mtime
    except OSError:
        return {}
    cached = _FORCE_WEIGHT_INDEX.get(root)
    if not force and cached is not None and cached[0] == stamp:
        return cached[1]

    index: Dict[str, str] = {}
    try:
        names = sorted(os.listdir(root))
    except OSError:
        names = []
    for name in names:
        sn_dir = os.path.join(root, name)
        if not os.path.isdir(sn_dir):
            continue
        try:
            files = sorted(os.listdir(sn_dir))
        except OSError:
            continue
        picked = None
        for backend in FORCE_WEIGHT_BACKENDS:
            for suffix in (backend + ".enc", backend):
                cand = os.path.join(sn_dir, name + suffix)
                if os.path.isfile(cand) and _force_weight_usable(cand):
                    picked = cand
                    break
            if picked:
                break
        if picked is None:  # 文件名与 SN 不一致时，取该目录里此后端能用的第一个
            for backend in FORCE_WEIGHT_BACKENDS:
                cands = [os.path.join(sn_dir, f) for f in files
                         if f.lower().endswith((backend, backend + ".enc"))]
                cands = [c for c in cands if _force_weight_usable(c)]
                if cands:
                    picked = cands[0]
                    break
        if picked:
            index[name] = picked
            index.setdefault(name.upper(), picked)
    _FORCE_WEIGHT_INDEX[root] = (stamp, index)
    return index


def _force_weight_for_sn(sn: str) -> Optional[str]:
    """按 SN 在 LM_sensor_force_weight/ 里找权重文件（找不到返回 None）。"""
    if not sn:
        return None
    for root in _force_weight_dirs():
        index = _scan_force_weight_dir(root)
        path = index.get(sn) or index.get(sn.upper())
        if not path:  # 没命中就忽略缓存重扫一次：权重复制进行中/后加的 SN 也能认出来
            index = _scan_force_weight_dir(root, force=True)
            path = index.get(sn) or index.get(sn.upper())
        if path:
            return path
    return None


def _force_model_tag(path: Optional[str]) -> str:
    """日志用：力模型的来源标签（只可能来自厂商按 SN 下发的权重目录）。"""
    if not path:
        return "无"
    ap = os.path.abspath(path)
    for root in _force_weight_dirs():
        if ap.startswith(os.path.abspath(root) + os.sep):
            return "LM_sensor_force_weight"
    return "其它路径"


def _resolve_force_model_path(sn: str) -> Optional[str]:
    """解析某传感器的力模型路径：只认厂商按 SN 下发的 LM_sensor_force_weight/<SN>/ 权重。

    取不到就返回 None，调用方应据此**不要**请求 6 维力（只记形变），
    而不是硬请求让它整帧读失败（SDK 请求 FORCE6D_VECTOR 时必须已配置力模型）。
    """
    return _force_weight_for_sn(sn)


def _force_data_types():
    """返回本 SDK 版本可用的 (6D 力, 滑动状态) 数据类型；老版本缺失的为 None。"""
    try:
        from pyvitaisdk import VTSDataType as D
    except Exception:  # noqa: BLE001
        return None, None
    return getattr(D, "FORCE6D_VECTOR", None), getattr(D, "SLIP_STATE", None)


def _force6d_vector(value) -> np.ndarray:
    """把 SDK 返回的 FORCE6D_VECTOR 规整成 (6,) float32 = [Fx,Fy,Fz,Tx,Ty,Tz]。

    兼容 (6,) 与 (N,M,6)（后者对前两维取均值），取不到时返回全 NaN。
    """
    out = np.full(6, np.nan, dtype=np.float32)
    if value is None:
        return out
    a = np.asarray(value, dtype=np.float32)
    if a.size == 0:
        return out
    if a.ndim >= 2 and a.shape[-1] >= 6:
        return a[..., :6].reshape(-1, 6).mean(axis=0).astype(np.float32)
    flat = a.reshape(-1)
    out[:min(6, flat.size)] = flat[:min(6, flat.size)]
    return out


# --------------------------------------------------------------------------- #
# 采集（原生内嵌）：跑在后台线程
# --------------------------------------------------------------------------- #


def _raw_to_bgr(img):
    """把任意原始帧转成可写 mp4 的 uint8 BGR (H,W,3)；不认识的格式返回 None。"""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if a.ndim == 3:
        c = a.shape[2]
        if c == 1:
            return cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2BGR)
        if c == 3:
            return a.copy()
        if c == 4:
            return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    return None


def _sanitize_name(name, default):
    """清洗用户输入的目录/文件名：去路径分隔符与非法字符，返回安全单级名。"""
    s = (name or "").strip().replace("\\", "/").strip("/").replace("/", "_")
    for ch in '<>:"|?*':
        s = s.replace(ch, "_")
    return s.strip(" .") or default


def _next_episode_index(task):
    """扫描 data/<task>/ 下已有的 episode_* 目录，返回下一个索引（从 0 开始）。"""
    base = os.path.join(DATA_DIR, task)
    if not os.path.isdir(base):
        return 0
    idxs = []
    for name in os.listdir(base):
        if name.startswith("episode_") and os.path.isdir(os.path.join(base, name)):
            try:
                idxs.append(int(name.split("_", 1)[1]))
            except ValueError:
                pass
    return (max(idxs) + 1) if idxs else 0


class DetectWorker(QThread):
    """只做设备枚举（VTSDeviceFinder().get_devices()），不标定，速度很快。

    信号：
        devices_found(list)    SN 列表（str）
        status(str)            状态文本
    """
    devices_found = pyqtSignal(list)
    status = pyqtSignal(str)

    def run(self):
        from pyvitaisdk import VTSDeviceFinder, VTSError
        try:
            finder = VTSDeviceFinder()
            devices = finder.get_devices()
        except VTSError as e:
            self.status.emit(tr("detect_fail", e=e, s=getattr(e, 'suggestion', '')))
            return
        sns = [d.SN for d in devices]
        self.devices_found.emit(sns)
        self.status.emit(tr("detect_found", n=len(sns)))


# --------------------------------------------------------------------------- #
# 分段写盘与采集
# --------------------------------------------------------------------------- #
SEGMENT_FRAMES = 9000   # 分段帧数（≈30fps × 5 分钟）；所有数据类型的段边界按帧号对齐
# 可勾选的保存类型：warped / depth_map / marker_current / force / slip / diff 视频 / raw 原生视频
SAVE_TYPES = ("warped", "depth", "marker", "force", "slip", "diff", "raw")
# 预览单格（4 面板并排）发给 GUI 前的最大宽度；每面板约 TILE_MAX_WIDTH/4
TILE_MAX_WIDTH = 960


def _npy_header(shape, descr, total=None):
    """构造 npy v1.0 头部字节；total 指定时补齐到该总长度（段尾按实际帧数重写同长头用）。"""
    d = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (descr, repr(tuple(shape)))
    base = 10 + len(d) + 1   # magic(6) + ver(2) + hlen(2) + dict + '\n'
    pad = ((64 - base % 64) % 64) if total is None else (total - base)
    if pad < 0:
        raise ValueError("npy header target length too small")
    s = d + " " * pad + "\n"
    return b"\x93NUMPY\x01\x00" + (len(s)).to_bytes(2, "little") + s.encode("latin1")


class _NpySegWriter:
    """大数组 npy 段的增量写：开头按 SEGMENT_FRAMES 写头，逐帧 tofile 追加（内存零累积），
    收尾按实际帧数重写同长头并截断。帧数据必须定形状定 dtype。"""

    def __init__(self, path, frame_shape, descr):
        self._shape = tuple(frame_shape)
        self._descr = descr
        self._count = 0
        self._fb = int(np.prod(self._shape)) * np.dtype(descr).itemsize
        self._f = open(path, "wb")
        h = _npy_header((SEGMENT_FRAMES,) + self._shape, descr)
        self._hlen = len(h)
        self._f.write(h)

    @property
    def count(self):
        return self._count

    def append(self, arr):
        a = np.ascontiguousarray(arr)
        if a.shape != self._shape:
            raise ValueError(f"frame shape {a.shape} != {self._shape}")
        self._f.write(a.tobytes())
        self._count += 1

    def close(self):
        if self._f is None:
            return
        h = _npy_header((self._count,) + self._shape, self._descr, total=self._hlen)
        self._f.flush()
        self._f.seek(0)
        self._f.write(h)
        self._f.truncate(self._hlen + self._count * self._fb)
        self._f.close()
        self._f = None


class SegmentRecorder:
    """单传感器、单 episode 的分段写盘器（独立写线程，采集线程只入队，队列满丢帧计数）。

    入队 bundle: {"idx", "warped", "diff", "raw", "depth", "marker", "force", "slip"}（值可为 None）。
    帧号不连续（采集丢帧/队列丢帧）时自动补位：depth/marker/force 补 NaN、slip 补 -1、
    视频（warped/diff/raw）重复上一帧，保证「段内索引 == episode 全局帧号 - 段起始帧」。
    """

    def __init__(self, episode_dir, sn, save, fps):
        self._dir = os.path.join(episode_dir, sn)
        self.save_types = set(save)
        self._fps = max(float(fps), 5.0)
        self.dropped = 0
        self._next = 0                       # 期望写入的下一帧号
        self._seg = None
        self._q = queue.Queue(maxsize=180)   # ~6s 缓冲（深度帧 1.2MB × 180 ≈ 220MB/传感器）
        self._vw = {"warped": None, "diff": None, "raw": None}
        self._last_vis = {"warped": None, "diff": None, "raw": None}
        self._counts = {"warped": 0, "diff": 0, "raw": 0}
        self._depth = None
        self._depth_shape = None
        self._marker_shape = None
        self._nan = {"force": np.full(6, np.nan, np.float32)}
        self._small = {"marker": [], "force": [], "slip": []}
        self._manifest = {}
        for kind in self.save_types:
            os.makedirs(os.path.join(self._dir, kind), exist_ok=True)
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    # -- 采集线程 --
    def enqueue(self, bundle):
        try:
            self._q.put_nowait(bundle)
        except queue.Full:
            self.dropped += 1

    def close(self):
        self._q.put(None)
        self._th.join()

    def manifest(self):
        return self._manifest

    # -- 写线程 --
    def _run(self):
        while True:
            b = self._q.get()
            if b is None:
                break
            try:
                self._handle(b)
            except Exception:  # noqa: BLE001 写盘异常不拖垮采集
                pass
        self._close_segment()

    def _handle(self, b):
        idx = int(b["idx"])
        if idx < self._next:
            return
        while self._next < idx:          # 缺口补位
            self._write_frame(self._next, None)
            self._next += 1
        self._write_frame(idx, b)
        self._next = idx + 1

    def _write_frame(self, idx, b):
        seg = (idx // SEGMENT_FRAMES) * SEGMENT_FRAMES
        if self._seg != seg:
            self._close_segment()
            self._open_segment(seg)

        def val(k):
            return None if b is None else b.get(k)

        # 视频类（warped / diff / raw）：缺口重复上一帧；首帧到达时回填段内欠帧
        for kind in ("warped", "diff", "raw"):
            if kind not in self.save_types:
                continue
            img = val(kind)
            bgr = _raw_to_bgr(img) if img is not None else None
            w = self._vw[kind]
            if w is None and bgr is not None:
                h, wd = bgr.shape[:2]
                path = os.path.join(self._dir, kind, f"segment_{seg:06d}.mp4")
                cand = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                       self._fps, (wd, h))
                if cand.isOpened():
                    w = self._vw[kind] = cand
                    for _ in range(idx - seg):
                        w.write(bgr)
                        self._counts[kind] += 1
                else:
                    cand.release()
            if w is not None:
                use = bgr if bgr is not None else self._last_vis[kind]
                if use is not None:
                    w.write(use)
                    self._counts[kind] += 1
            if bgr is not None:
                self._last_vis[kind] = bgr

        # depth：float32 原始深度图，段内逐帧增量写，缺口补 NaN
        if "depth" in self.save_types:
            arr = val("depth")
            if arr is not None and self._depth_shape is None:
                self._depth_shape = tuple(np.asarray(arr).shape)
                self._nan["depth"] = np.full(self._depth_shape, np.nan, np.float32)
            if self._depth is None and self._depth_shape is not None:
                self._depth = _NpySegWriter(
                    os.path.join(self._dir, "depth", f"segment_{seg:06d}.npy"),
                    self._depth_shape, "<f4")
                for _ in range(idx - seg):   # shape 未定时欠下的帧补 NaN
                    self._depth.append(self._nan["depth"])
            if self._depth is not None:
                self._depth.append(np.asarray(
                    arr if arr is not None else self._nan["depth"], np.float32))

        # marker_current：小数组攒段内 list，缺口补 NaN（shape 未定时先欠着）
        if "marker" in self.save_types:
            arr = val("marker")
            if arr is not None and self._marker_shape is None:
                self._marker_shape = tuple(np.asarray(arr).shape)
                self._nan["marker"] = np.full(self._marker_shape, np.nan, np.float32)
                self._small["marker"].extend(
                    [self._nan["marker"]] * (idx - seg - len(self._small["marker"])))
            if self._marker_shape is not None:
                self._small["marker"].append(np.asarray(
                    arr if arr is not None else self._nan["marker"], np.float32))

        # force / slip：形状固定，直接补位
        if "force" in self.save_types:
            fv = val("force")
            self._small["force"].append(np.asarray(
                fv if fv is not None else self._nan["force"], np.float32))
        if "slip" in self.save_types:
            sv = val("slip")
            self._small["slip"].append(-1 if sv is None else int(sv))

    def _open_segment(self, seg):
        self._seg = seg
        self._counts = {"warped": 0, "diff": 0, "raw": 0}
        self._vw = {"warped": None, "diff": None, "raw": None}
        self._depth = None
        self._small = {"marker": [], "force": [], "slip": []}

    def _close_segment(self):
        if self._seg is None:
            return
        seg = self._seg
        for kind in ("warped", "diff", "raw"):
            w = self._vw.get(kind)
            if w is not None:
                w.release()
                self._manifest.setdefault(kind, []).append(
                    {"file": f"{kind}/segment_{seg:06d}.mp4",
                     "start_frame": seg, "frames": self._counts[kind]})
                self._vw[kind] = None
        if self._depth is not None:
            self._depth.close()
            self._manifest.setdefault("depth", []).append(
                {"file": f"depth/segment_{seg:06d}.npy",
                 "start_frame": seg, "frames": self._depth.count})
            self._depth = None
        for kind in ("marker", "force", "slip"):
            buf = self._small.get(kind)
            if buf:
                np.save(os.path.join(self._dir, kind, f"segment_{seg:06d}.npy"),
                        np.stack(buf))
                self._manifest.setdefault(kind, []).append(
                    {"file": f"{kind}/segment_{seg:06d}.npy",
                     "start_frame": seg, "frames": len(buf)})
        self._small = {"marker": [], "force": [], "slip": []}
        self._seg = None


class _SharedSensor:
    """把渲染线程的 SensorWorker 适配成采集线程眼中的「传感器」。

    同一台相机不能被两个线程同时打开，因此采集线程不再自己开设备，而是
    复用渲染线程已经打开的那台：collect 直接返回渲染线程的最新一帧。
    """

    def __init__(self, worker):
        self._worker = worker

    def collect_sensor_data(self, *types):
        # types 只为兼容原调用签名；渲染线程已按全量类型采集
        return self._worker.latest_data()

    def calibrate(self):
        # 转发给渲染线程：由它重新标定，采集端基线随后重建
        self._worker.calibrate()

    def release(self):
        # 设备归渲染线程所有，采集结束不释放
        pass


class DataCaptureWorker(QThread):
    """数据采集：标定 + 出流 + 按「开始保存 / 停止保存」录制 episode（分段落盘）。

    保存内容由 set_spec 注入的 save 集合决定（warped/diff/depth/marker/force/slip/raw），
    决定的是「落盘哪些类型」；预览所需的 warped/diff/marker 图/深度图每帧恒取（见 _make_tile），
    marker 坐标 / 6 维力 / 滑动状态才按勾选追加请求。产物结构：
        data/<task>/episode_<N>/frames_index.csv   逐帧时间轴总表
        data/<task>/episode_<N>/meta.json
        data/<task>/episode_<N>/<sn>/{bg.png, depth_baseline.npy, marker_baseline.npy}
        data/<task>/episode_<N>/<sn>/<type>/segment_XXXXXX.{mp4,npy}

    信号：
        status(str)
        frame(dict)                {"tiles": {sn: BGR}, "stats": {sn: {...}}, "labels": {sn: str}}
        streaming(bool)            开始/结束持续出流
        episode_changed(bool, str) episode 开始/停止 + 描述
    """
    status = pyqtSignal(str)
    frame = pyqtSignal(object)
    streaming = pyqtSignal(bool)
    episode_changed = pyqtSignal(bool, str)

    def __init__(self, sensors_sn, labels, sensor_workers=None):
        super().__init__()
        self._sensors_sn = list(sensors_sn)
        self._labels = list(labels)
        # sn -> SensorWorker：共享渲染线程已打开的设备；None 表示自己打开（独立模式）
        self._sensor_workers = dict(sensor_workers) if sensor_workers else None
        self._last_ts = {}      # sn -> 上次落盘帧的时间戳（共享模式下用于去重）
        self._last_frames = {}  # sn -> 最近一次解析后的帧数据
        self._lock = threading.Lock()
        self._spec = None
        self._episode_start = False   # 面板请求开始一个 episode
        self._episode_stop = False    # 面板请求结束当前 episode
        self._recording = False       # 当前是否有 episode 进行中
        self._recalibrate = False     # 面板请求重新标定 + 重建基线
        self._discard = False         # 面板请求丢弃当前正在录制的 episode（停止时不落盘）
        self._active = threading.Event()
        self._active.set()

    # -- 主线程可调（线程安全） --
    def request_stop(self):
        self._active.clear()

    def set_spec(self, spec):
        with self._lock:
            self._spec = spec

    def start_episode(self):
        with self._lock:
            self._episode_start = True

    def stop_episode(self):
        with self._lock:
            self._episode_stop = True

    def request_recalibrate(self):
        with self._lock:
            self._recalibrate = True

    def episode_active(self) -> bool:
        with self._lock:
            return self._recording

    def discard_episode(self):
        """请求丢弃当前正在录制的 episode：停止采集时不再落盘，直接删除。

        仅置位标志，由 run() 的 finally 在退出时删除未完成的 episode 目录。
        """
        with self._lock:
            self._discard = True

    # -- worker 线程 --
    def run(self):
        from pyvitaisdk import VTSensor, VTSDeviceFinder, VTSDataType, VTSError

        sensors = []  # [(sn, sensor), ...]
        force_models = {}   # SN -> 力模型路径（None = 该传感器只记形变，不请求 6 维力）
        rec = None
        failed = False
        try:
            self.status.emit(tr("cap_opening", n=len(self._sensors_sn)))
            sensor_labels = []
            # 6 维力：SDK 要求构造时显式给力模型，否则请求 FORCE6D_VECTOR 直接报错。
            force_type, slip_type = _force_data_types()
            if self._sensor_workers is not None:
                # 共享模式：复用渲染线程已经打开的设备，不再二次打开相机
                for sn, lbl in zip(self._sensors_sn, self._labels):
                    worker = self._sensor_workers.get(sn)
                    if worker is None or not worker.isRunning():
                        self.status.emit(tr("cap_not_found", sn=sn, lbl=lbl))
                        continue
                    sensors.append((sn, _SharedSensor(worker)))
                    sensor_labels.append(lbl)
                    force_models[sn] = "shared"
                    self.status.emit(tr("cap_opened", sn=sn))
            else:
                # 独立模式：自己打开并标定设备（standalone GUI 使用）
                finder = VTSDeviceFinder()
                for sn, lbl in zip(self._sensors_sn, self._labels):
                    dev = finder.get_device_by_sn(sn)
                    if dev is None:
                        self.status.emit(tr("cap_not_found", sn=sn, lbl=lbl))
                        continue
                    self.status.emit(tr("cap_calibrating", sn=sn, lbl=lbl))
                    force_model_path = _resolve_force_model_path(sn) if force_type else None
                    try:
                        sensor = None
                        if force_model_path:
                            try:  # 力模型坏掉不该阻断采集：退回「无力模型」再开一次
                                sensor = VTSensor(config=dev, force_model_path=force_model_path)
                            except Exception as e:  # noqa: BLE001
                                self.status.emit(tr("cap_force_fail", sn=sn, e=e))
                                force_model_path = None
                        if sensor is None:
                            sensor = VTSensor(config=dev)
                        sensor.calibrate()
                        sensors.append((sn, sensor))
                        sensor_labels.append(lbl)
                        force_models[sn] = force_model_path
                        self.status.emit(tr("cap_opened", sn=sn))
                        if force_model_path:
                            self.status.emit(tr("cap_force_model", sn=sn,
                                                name=os.path.basename(force_model_path),
                                                src=_force_model_tag(force_model_path)))
                    except Exception as e:  # noqa: BLE001 单设备失败不影响其它
                        self.status.emit(tr("cap_open_fail", sn=sn, e=e))
            if not sensors:
                self.status.emit(tr("cap_none"))
                return

            sn_list = [sn for sn, _s in sensors]
            # 预览 4 面板（见 _make_tile）恒需的数据类型；「保存内容」勾选只决定是否落盘
            PREVIEW_DTS = (VTSDataType.TIME_STAMP, VTSDataType.RAW_IMG, VTSDataType.WARPED_IMG,
                           VTSDataType.DIFF_IMG, VTSDataType.DEPTH_MAP, VTSDataType.MARKER_IMG)
            DEPTH_VMAX = 0.3                # 预览上色量程（NOISE_FLOOR 用模块级常量）
            statics = {}          # SN -> {"bg", "depth_base", "marker_base"}（出流首帧捕获）
            slip_bad = set()      # 请求 SLIP_STATE 失败过的 SN（该台本次出流不再请求 slip）
            start_t = time.monotonic()
            frame_count = 0
            fps_est = 30.0
            self.streaming.emit(True)
            self.status.emit(tr("cap_streaming", n=len(sensors)))
            tick = 0

            while self._active.is_set():
                with self._lock:
                    spec = self._spec or {}
                    save = set(spec.get("save") or ())
                    ep_start = self._episode_start
                    ep_stop = self._episode_stop
                    self._episode_start = False
                    self._episode_stop = False
                    recalib = self._recalibrate
                    self._recalibrate = False

                # 重新标定：SDK 重设背景，基线/背景图下一帧重建
                if recalib:
                    for _sn, sensor in sensors:
                        try:
                            sensor.calibrate()
                        except Exception:  # noqa: BLE001
                            pass
                    statics = {}
                    self.status.emit(tr("recalibrated"))

                # 本帧要取的数据类型：预览所需的 warped/diff/marker 图/深度图恒取，
                # 其余（marker 坐标 / 6 维力 / 滑动状态）按「保存内容」勾选追加
                dts = list(PREVIEW_DTS)
                if "marker" in save:
                    dts.append(VTSDataType.MARKER_CURRENT_VECTOR)
                    if any("marker_base" not in (statics.get(sn) or {}) for sn in sn_list):
                        dts.append(VTSDataType.MARKER_ORIGIN_VECTOR)
                want_force = "force" in save

                frames = {}   # SN -> 本帧数据 dict
                updated = False
                for sn, sensor in sensors:
                    try:
                        types = list(dts)
                        # 力与滑动检测互相独立：力要力模型，slip 只要有 SLIP_STATE 类型就请求
                        use_force = (want_force and force_type is not None
                                     and bool(force_models.get(sn)))
                        use_slip = ("slip" in save and slip_type is not None
                                    and sn not in slip_bad)
                        if use_force:
                            types.append(force_type)
                        if use_slip:
                            types.append(slip_type)
                        try:
                            data = sensor.collect_sensor_data(*types)
                        except VTSError as e:
                            if not (use_force or use_slip):
                                raise
                            # 附加数据出问题就永久退回基础数据，不让整帧读失败
                            if use_force:
                                force_models[sn] = None
                                self.status.emit(tr("cap_force_fail", sn=sn, e=e))
                            if use_slip:
                                slip_bad.add(sn)
                                self.status.emit(tr("cap_slip_fail", sn=sn, e=e))
                            data = sensor.collect_sensor_data(*dts)
                            use_force = use_slip = False
                        ts = data.get(VTSDataType.TIME_STAMP)
                        # 共享模式下 data 是渲染线程的最新帧：同一帧不重复落盘
                        if ts is None or (sn in self._last_ts and self._last_ts[sn] == ts):
                            if sn in self._last_frames:
                                frames[sn] = self._last_frames[sn]
                            continue
                        self._last_ts[sn] = ts
                        slip_val = None
                        if use_slip:
                            sv = data.get(slip_type)
                            slip_val = None if sv is None else int(sv)
                        frames[sn] = {
                            "ts": ts,
                            "raw": data.get(VTSDataType.RAW_IMG),
                            "warped": data.get(VTSDataType.WARPED_IMG),
                            "diff": data.get(VTSDataType.DIFF_IMG),
                            "marker_img": data.get(VTSDataType.MARKER_IMG),
                            "depth": data.get(VTSDataType.DEPTH_MAP),
                            "marker": data.get(VTSDataType.MARKER_CURRENT_VECTOR),
                            "force": (_force6d_vector(data.get(force_type))
                                      if use_force else None),
                            "slip": slip_val,
                        }
                        self._last_frames[sn] = frames[sn]
                        updated = True
                        # 静态基线捕获：出流首帧（或重标定后首帧）
                        st = statics.setdefault(sn, {})
                        if "bg" not in st:
                            img = frames[sn]["warped"]
                            if img is None:
                                img = frames[sn]["raw"]
                            if img is not None:
                                st["bg"] = np.asarray(img).copy()
                        if "depth_base" not in st and frames[sn]["depth"] is not None:
                            st["depth_base"] = np.asarray(frames[sn]["depth"], np.float32).copy()
                            d = st["depth_base"]
                            self.status.emit(tr(
                                "cap_baseline", sn=sn,
                                min=f"{np.nanmin(d):.4f}",
                                max=f"{np.nanmax(d):.4f}",
                                mean=f"{np.nanmean(d):.4f}"))
                        if "marker_base" not in st:
                            org = data.get(VTSDataType.MARKER_ORIGIN_VECTOR)
                            if org is not None:
                                st["marker_base"] = np.asarray(org, np.float32).copy()
                    except VTSError as e:
                        self.status.emit(tr("cap_read_err", sn=sn, e=e))

                # episode 状态机：「开始保存 / 停止保存」是录制的唯一驱动
                if ep_start and rec is None:
                    rec = self._begin_episode(sensors, sn_list, sensor_labels,
                                              force_models, slip_bad, statics, save, fps_est)
                    if rec is not None:
                        with self._lock:
                            self._recording = True
                        self.episode_changed.emit(True, rec["describe"])
                        self.status.emit(tr("cap_episode_start", desc=rec["describe"]))
                elif ep_stop and rec is not None:
                    self._end_episode(rec)
                    rec = None
                    with self._lock:
                        self._recording = False
                    self.episode_changed.emit(False, "")

                # 本轮没有任何新帧（共享模式下渲染线程尚未更新）→ 不重复落盘/预览
                if not updated:
                    self.msleep(4)
                    continue

                # 录制本帧：帧号即 frames_index 行号 == 各段内全局帧号
                if rec is not None:
                    idx = rec["next_frame"]
                    for sn, r in rec["recorders"].items():
                        f = frames.get(sn) or {}
                        r.enqueue({"idx": idx, "warped": f.get("warped"),
                                   "diff": f.get("diff"), "raw": f.get("raw"),
                                   "depth": f.get("depth"),
                                   "marker": f.get("marker"), "force": f.get("force"),
                                   "slip": f.get("slip")})
                    wall = datetime.now()
                    csv.writer(rec["index_f"]).writerow(
                        [idx, wall.strftime("%Y-%m-%dT%H:%M:%S.%f")]
                        + [int((frames.get(sn) or {}).get("ts") or 0) for sn in sn_list])
                    rec["next_frame"] += 1

                # 预览（隔帧发，减轻 GUI 负担）
                tick += 1
                if tick % 2 == 0:
                    tiles, stats = {}, {}
                    for sn in sn_list:
                        f = frames.get(sn)
                        if not f:
                            continue
                        tile = self._make_tile(f, statics.get(sn) or {},
                                               NOISE_FLOOR, DEPTH_VMAX)
                        if tile is not None:
                            tiles[sn] = tile
                        fv = f.get("force")
                        stats[sn] = {
                            "slip": f.get("slip"),
                            "fabs": (float(np.linalg.norm(fv[:3]))
                                     if fv is not None and not np.isnan(fv[0]) else None),
                        }
                    if tiles:
                        self.frame.emit({"tiles": tiles, "stats": stats,
                                         "labels": dict(zip(sn_list, sensor_labels))})

                frame_count += 1
                fps_est = frame_count / max(time.monotonic() - start_t, 1e-6)
                self.msleep(1)
        except Exception as e:  # noqa: BLE001 让异常原因可见，而不是静默结束线程
            failed = True
            self.status.emit(tr("cap_fatal", e=e))
        finally:
            # 收尾：仍在录制时按「保存 / 丢弃」二选一处理
            try:
                if rec is not None:
                    if self._discard:
                        # 丢弃：先关闭写线程（让其自然退出），再删除整个 episode 目录
                        for r in rec["recorders"].values():
                            r.close()
                        try:
                            rec["index_f"].close()
                        except Exception:  # noqa: BLE001
                            pass
                        shutil.rmtree(rec["episode_dir"], ignore_errors=True)
                        self.status.emit(tr("episode_discarded", desc=rec["episode_dir"]))
                    else:
                        self._end_episode(rec)
            except Exception:  # noqa: BLE001
                pass
            with self._lock:
                self._recording = False
            for _sn, sensor in sensors:
                try:
                    sensor.release()
                except Exception:  # noqa: BLE001
                    pass
            self.streaming.emit(False)
            if not failed:
                self.status.emit(tr("cap_stopped"))

    @staticmethod
    def _make_tile(f, st, noise_floor, depth_vmax):
        """拼单传感器预览格：warped + diff + marker 图 + 深度形变伪彩，四面板并排。

        主画面 warped 拿不到时退回 raw；深度面板画相对首帧基线的形变，
        |形变| < noise_floor 视为噪声。
        """
        primary = f.get("warped")
        primary = _raw_to_bgr(primary) if primary is not None else _raw_to_bgr(f.get("raw"))
        if primary is None:
            return None
        h, w = primary.shape[:2]
        parts = [primary]
        # diff / marker 面板：SDK 直接给的整幅图；尺寸与主画面不一致时先对齐再并排
        for img in (f.get("diff"), f.get("marker_img")):
            bgr = _raw_to_bgr(img) if img is not None else None
            if bgr is None:
                continue
            if bgr.shape[:2] != (h, w):
                bgr = cv2.resize(bgr, (w, h))
            parts.append(bgr)
        # 深度面板：相对基线的形变伪彩
        d = f.get("depth")
        if d is not None:
            rel = np.asarray(d, np.float32)
            base = st.get("depth_base")
            if base is not None:
                rel = rel - base
                rel[np.abs(rel) < noise_floor] = 0.0
            tmp = np.clip(np.abs(rel) / depth_vmax * 255, 0, 255).astype(np.uint8)
            cm = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)
            if cm.shape[:2] != (h, w):
                cm = cv2.resize(cm, (w, h))
            parts.append(cm)
        tile = np.hstack(parts) if len(parts) > 1 else parts[0]
        h, w = tile.shape[:2]
        if w > TILE_MAX_WIDTH:   # 发给 GUI 前降尺寸，减轻主线程开销
            tile = cv2.resize(tile, (TILE_MAX_WIDTH, int(h * TILE_MAX_WIDTH / w)))
        return tile

    # -- episode 录制会话 --
    def _begin_episode(self, sensors, sn_list, sensor_labels, force_models, slip_bad, statics,
                       save, fps_est):
        spec = self._spec or {}
        task = _sanitize_name(spec.get("task"), "unnamed")
        save = set(save)
        if not save:
            self.status.emit(tr("start_no_type_msg"))
            return None
        idx = _next_episode_index(task)
        episode_dir = os.path.join(DATA_DIR, task, f"episode_{idx}")
        os.makedirs(episode_dir, exist_ok=True)

        recorders = {}
        for sn in sn_list:
            ssave = set(save)
            if not force_models.get(sn):
                ssave.discard("force")       # 无力模型：该台不落地 force
            if sn in slip_bad:
                ssave.discard("slip")        # 该台 SLIP_STATE 请求不通：不落地 slip
            if not ssave:
                continue
            recorders[sn] = SegmentRecorder(episode_dir, sn, ssave, fps_est)
        if not recorders:
            self.status.emit(tr("start_no_type_msg"))
            return None

        # 静态小文件：随对应类型自动保存（出流首帧捕获，重标定后重建）
        for sn in sn_list:
            st = statics.get(sn) or {}
            d = os.path.join(episode_dir, sn)
            os.makedirs(d, exist_ok=True)
            if "warped" in save and st.get("bg") is not None:
                cv2.imwrite(os.path.join(d, "bg.png"), st["bg"])
            if "depth" in save and st.get("depth_base") is not None:
                np.save(os.path.join(d, "depth_baseline.npy"), st["depth_base"])
            if "marker" in save and st.get("marker_base") is not None:
                np.save(os.path.join(d, "marker_baseline.npy"), st["marker_base"])

        index_f = open(os.path.join(episode_dir, "frames_index.csv"), "w", newline="")
        csv.writer(index_f).writerow(
            ["frame", "wall_time"] + [f"{sn}_ms" for sn in sn_list])
        rec = {
            "task": task,
            "episode_index": idx,
            "episode_dir": episode_dir,
            "start_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "save": save,
            "sensors": list(sensors),
            "sensor_labels": list(sensor_labels),
            "force_models": dict(force_models or {}),
            "recorders": recorders,
            "index_f": index_f,
            "next_frame": 0,
            "describe": episode_dir,
        }
        self._write_episode_meta(rec, "recording")
        return rec

    def _write_episode_meta(self, rec, status):
        sensors_meta = []
        for lbl, (sn, _s) in zip(rec["sensor_labels"], rec["sensors"]):
            r = rec["recorders"].get(sn)
            sensors_meta.append({
                "serial_number": sn,
                "label": lbl,
                "force_model": (rec.get("force_models") or {}).get(sn),
                "save_types": sorted(r.save_types) if r is not None else [],
                "dropped_frames": r.dropped if r is not None else 0,
            })
        meta = {
            "recording": {"start_time": rec["start_time"], "status": status},
            "session": {
                "task": rec["task"],
                "episode_index": rec["episode_index"],
                "sensor_count": len(rec["sensors"]),
                "save_types": sorted(rec["save"]),
                "segment_frames": SEGMENT_FRAMES,
                "frames": rec["next_frame"],
            },
            "sensors": sensors_meta,
            "segments": {sn: r.manifest() for sn, r in rec["recorders"].items()},
            "static_files": {
                "bg": "bg.png — 出流首帧 warped（勾 warped 时）",
                "depth_baseline": "depth_baseline.npy (H,W) float32 — 出流首帧 depth_map（勾 depth 时）",
                "marker_baseline": "marker_baseline.npy (N,M,2) float32 — marker 初始位置（勾 marker 时）",
            },
            "alignment": (
                "段内索引 + 段起始帧 == frames_index.csv 的 frame 列 == 视频段内帧号；"
                "某传感器某帧缺失时 depth/marker/force 为 NaN、slip 为 -1、视频重复上一帧"
            ),
            "timestamps": "frames_index.csv: frame, wall_time（电脑时间）, <sn>_ms（设备毫秒时间戳）",
            "slip_state": SLIP_STATE_DOC,
            "dtypes": {
                "depth": "float32 (T,H,W) 原始 depth_map（未减基线）",
                "marker_current": "float32 (T,N,M,2) 去畸变图 marker 像素坐标；位移 = current - marker_baseline",
                "force": "float32 (T,6) [Fx,Fy,Fz,Tx,Ty,Tz]，量纲由力模型 normalization_params.json 决定",
                "slip": "int8 (T,)，见 slip_state",
            },
        }
        try:
            with open(os.path.join(rec["episode_dir"], "meta.json"),
                      "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=4)
        except Exception:  # noqa: BLE001
            pass

    def _end_episode(self, rec):
        for r in rec["recorders"].values():
            r.close()
        rec["index_f"].close()
        self._write_episode_meta(rec, "completed")
        total = 0
        for root, _dirs, files in os.walk(rec["episode_dir"]):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        dropped = sum(r.dropped for r in rec["recorders"].values())
        self.status.emit(tr("episode_ended_detail", frames=rec["next_frame"],
                            size=f"{total / 1e6:.1f}", dropped=dropped))


class FingerAssignWidget(QWidget):
    """左右手各 5 指 ↔ SN 下拉框组（全局互斥），配置持久化到 config/sensor_fingers.json。

    左右手两组并排，各占一半宽度，方便横向铺满控制栏。
    """

    def __init__(self):
        super().__init__()
        self._sns = []
        self._desired = {lbl: None for lbl in _finger_labels()}  # label -> SN（None 未分配）
        self._combo = {}  # hand -> finger -> QComboBox
        self._hand_groups = []
        self._hand_box = {}    # hand -> QGroupBox（语言切换时刷新标题）
        self._finger_lbl = {}  # hand -> finger -> QLabel（语言切换时刷新文字）

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        for hand in HANDS:
            group = QGroupBox(tr("fingers_count", hand=tr(f"hand_{hand}"), n=len(FINGERS)))
            grid = QGridLayout(group)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(4)
            self._combo[hand] = {}
            self._finger_lbl[hand] = {}
            for j, finger in enumerate(FINGERS):
                lbl = QLabel(tr(f"finger_{finger}"))
                combo = QComboBox()
                combo.addItem(tr("unassigned"), None)
                combo.currentIndexChanged.connect(
                    lambda _idx, h=hand, f=finger: self._on_combo_changed(h, f))
                grid.addWidget(lbl, j, 0)
                grid.addWidget(combo, j, 1)
                self._combo[hand][finger] = combo
                self._finger_lbl[hand][finger] = lbl
            lay.addWidget(group, 1)
            self._hand_groups.append(group)
            self._hand_box[hand] = group

        self._load_config()
        _RETRANSLATE.append(self.retranslate)

    def retranslate(self):
        """语言切换时刷新标题、手指标签与「未分配」占位项。"""
        for hand in HANDS:
            box = self._hand_box.get(hand)
            if box is not None:
                box.setTitle(tr("fingers_count", hand=tr(f"hand_{hand}"), n=len(FINGERS)))
            for finger in FINGERS:
                lbl = self._finger_lbl.get(hand, {}).get(finger)
                if lbl is not None:
                    lbl.setText(tr(f"finger_{finger}"))
        self._refresh_combos()

    # -- 配置持久化 --
    def _load_config(self):
        cfg = {}
        if os.path.exists(CAPTURE_CONFIG_PATH):
            try:
                with open(CAPTURE_CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:  # noqa: BLE001
                cfg = {}
        if not isinstance(cfg, dict):
            return
        for hand in HANDS:
            hcfg = cfg.get(hand, {})
            if not isinstance(hcfg, dict):
                continue
            for finger in FINGERS:
                sn = hcfg.get(finger)
                if sn:
                    self._desired[f"{hand}_{finger}"] = sn

    def save_config(self):
        cfg = {"right": {}, "left": {}}
        for hand in HANDS:
            for finger in FINGERS:
                sn = self._desired.get(f"{hand}_{finger}")
                if sn:
                    cfg[hand][finger] = sn
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)   # config/ 不存在时自动创建
            with open(CAPTURE_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=4)
        except Exception:  # noqa: BLE001
            pass

    def set_sns(self, sns):
        self._sns = list(sns)
        self._refresh_combos()

    def set_enabled(self, on):
        for g in self._hand_groups:
            g.setEnabled(on)

    def collect_assign(self):
        """按采集顺序（右手拇指→小指、再左手）返回 (sensors_sn, labels)，只含已分配的手指。"""
        sensors_sn, labels = [], []
        for hand in HANDS:
            for finger in FINGERS:
                lbl = f"{hand}_{finger}"
                sn = self._desired.get(lbl)
                if sn:
                    sensors_sn.append(sn)
                    labels.append(lbl)
        return sensors_sn, labels

    def labels_for(self, sns):
        """返回与 sns 等长的标签列表：SN 被分配到某手指则用 手指标签，否则用 SN 本身。

        用于「采集所有连接到的传感器」时生成数据标签——手指标定仅作标签，
        不再限制采集范围（未做手指映射的已连接传感器直接以 SN 录制）。
        """
        sn_to_lbl = {sn: lbl for lbl, sn in self._desired.items() if sn}
        return [sn_to_lbl.get(sn, sn) for sn in sns]

    # -- 下拉框全局互斥 --
    def _refresh_combos(self):
        used = {sn for sn in self._desired.values() if sn}
        for hand in HANDS:
            for finger in FINGERS:
                lbl = f"{hand}_{finger}"
                combo = self._combo[hand][finger]
                cur = self._desired[lbl]
                combo.blockSignals(True)
                combo.clear()
                combo.addItem(tr("unassigned"), None)
                for sn in self._sns:
                    if sn == cur or sn not in used:
                        combo.addItem(sn, sn)
                if cur is not None and combo.findData(cur) >= 0:
                    combo.setCurrentIndex(combo.findData(cur))
                else:
                    # 已配置但设备未检测到：保留期望值（检测到后再自动匹配），下拉框暂显「未分配」
                    combo.setCurrentIndex(0)
                combo.blockSignals(False)

    def _on_combo_changed(self, hand, finger):
        lbl = f"{hand}_{finger}"
        sn = self._combo[hand][finger].currentData()
        if self._desired.get(lbl) == sn:
            return
        self._desired[lbl] = sn
        self._refresh_combos()
        self.save_config()

    def import_config(self, path):
        """从配置文件导入手指↔SN 映射并自动匹配下拉框。

        文件格式与 config/sensor_fingers.json 一致：
            {"right": {"thumb": "SN", "index": ...}, "left": {"thumb": "SN", ...}}
        返回 (是否成功, 说明文本)。
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:  # noqa: BLE001
            return False, tr("cfg_read_fail", e=e)
        if not isinstance(cfg, dict):
            return False, tr("cfg_format_err")

        applied = 0
        for hand in HANDS:
            hcfg = cfg.get(hand)
            if not isinstance(hcfg, dict):
                continue
            for finger in FINGERS:
                sn = hcfg.get(finger)
                if sn:
                    self._desired[f"{hand}_{finger}"] = str(sn)
                    applied += 1
        self._refresh_combos()
        self.save_config()

        detected = set(self._sns)
        matched = sum(1 for sn in self._desired.values() if sn and sn in detected)
        msg = tr("cfg_imported", applied=applied, matched=matched)
        if applied > matched:
            msg += tr("cfg_unmatched")
        return True, msg


class SensorTile(QWidget):
    """单个传感器的预览格：画面 + 标签/|F| 文字叠加 + 滑动状态色点；点击在放大/还原间切换。

    画面由采集线程隔帧推入（BGR ndarray），paintEvent 内等比缩放绘制。
    """

    clicked = pyqtSignal(str)

    # 滑动状态色点（与 SLIP_STATE_DOC 对应；未知/无数据显示灰点）
    _SLIP_COLOR = {1: "#4CAF50", 2: "#FFC107", 3: "#FF9800", 4: "#F44336",
                   5: "#9E9E9E", 6: "#2196F3"}

    def __init__(self, sn, parent=None):
        super().__init__(parent)
        self.sn = sn
        self._pix = None
        self._title = sn
        self._stats = {}
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_content(self, bgr, title, stats):
        """更新一格的画面与叠加信息；bgr 为 None 时保留上一帧画面。"""
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            self._pix = QPixmap.fromImage(qimg.copy())
        if title:
            self._title = title
        self._stats = stats or {}
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.sn)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 22))
        if self._pix is not None and not self._pix.isNull():
            scaled = self._pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
        st = self._stats
        info = self._title
        if st.get("fabs") is not None:
            info += f"  |F|={st['fabs']:.3f}"
        p.setPen(QColor(255, 255, 255))
        p.drawText(8, 18, info)
        slip = st.get("slip")
        if slip is not None:
            p.setBrush(QColor(self._SLIP_COLOR.get(slip, "#9E9E9E")))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(self.width() - 20, 8, 12, 12)
        p.end()


class CapturePanel(QWidget):
    """实时预览 + 数据采集面板。

    布局：左侧为控制栏（检测/采集按钮、手指标定、存储设置、任务录制），
          右侧为实时预览（占满剩余空间，随窗口缩放）。
    录制完全由「开始保存 / 停止保存」按钮驱动，每按一对 = 一个 episode_<N>。
    """

    # 请求主窗口路由回到实时查看器页
    back_to_viewer = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._worker = None
        self._detector = None
        self._starting = False   # 启动流程进行中（含基线确认弹窗）→ 防重复触发
        self._sns = []  # 检测到的 SN 列表
        self._finger_widget = FingerAssignWidget()

        outer = QHBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(20)

        # ---- 左侧：控制栏（包一层滚动区，窗口矮时可滚动） ----
        left = QVBoxLayout()
        left.setSpacing(10)

        self._title = QLabel(tr("app_title"))
        self._title.setObjectName("panelTitle")
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self.back_viewer_btn = QPushButton(tr("btn_back_viewer"))
        self.back_viewer_btn.setObjectName("langBtn")
        self.back_viewer_btn.setToolTip("返回实时查看器")
        self.back_viewer_btn.clicked.connect(self.back_to_viewer.emit)
        self.lang_btn = QPushButton(tr("lang_switch"))
        self.lang_btn.setObjectName("langBtn")
        self.lang_btn.setToolTip("Switch language / 切换语言")
        self.lang_btn.clicked.connect(self._toggle_lang)
        title_row.addWidget(self._title)
        title_row.addWidget(self.back_viewer_btn)
        title_row.addStretch(1)
        title_row.addWidget(self.lang_btn)
        self._sub = QLabel(tr("panel_subtitle"))
        self._sub.setObjectName("panelSub")
        self._sub.setWordWrap(True)
        left.addLayout(title_row)
        left.addWidget(self._sub)
        left.addSpacing(4)

        # 采集/停止按钮
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.detect_btn = QPushButton(tr("btn_detect"))
        self.import_btn = QPushButton(tr("btn_import"))
        self.start_btn = QPushButton(tr("btn_start"))
        self.stop_btn = QPushButton(tr("btn_stop"))
        self.recalib_btn = QPushButton(tr("btn_recalib"))
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.recalib_btn.setEnabled(False)   # 出流后才可重新标定
        self.detect_btn.clicked.connect(self._detect)
        self.import_btn.clicked.connect(self._import_config)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._shutdown)
        self.recalib_btn.clicked.connect(self._recalibrate)
        btn_row.addWidget(self.detect_btn)
        btn_row.addWidget(self.import_btn)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.recalib_btn)
        left.addLayout(btn_row)

        left.addWidget(self._finger_widget)

        # 存储设置
        self._store = QGroupBox(tr("store_title"))
        self._store_form = QFormLayout(self._store)
        self._store_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.task_edit = QLineEdit()
        self.task_edit.setPlaceholderText(tr("task_placeholder"))
        self.task_edit.textChanged.connect(self._refresh_path)
        self._store_form.addRow(tr("label_folder"), self.task_edit)
        self.path_label = QLabel("")
        self.path_label.setObjectName("panelSub")
        self.path_label.setWordWrap(True)
        self._store_form.addRow(tr("label_path"), self.path_label)
        left.addWidget(self._store)

        # 保存内容：每类数据一个复选框；采集线程按勾选向 SDK 请求数据类型（不勾不取不存）
        self._save_box = QGroupBox(tr("save_group"))
        save_col = QVBoxLayout(self._save_box)
        self._save_checks = {}
        for key in SAVE_TYPES:
            cb = ColorCheckBox(tr(f"save_{key}"))
            cb.setChecked(key not in ("diff", "raw"))   # 默认不勾 diff 视频与原生视频
            cb.toggled.connect(self._push_spec)
            self._save_checks[key] = cb
            save_col.addWidget(cb)
        left.addWidget(self._save_box)

        # 任务录制（episode）
        self._seg = QGroupBox(tr("seg_title"))
        seg_form = QFormLayout(self._seg)
        seg_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        seg_btn_row = QHBoxLayout()
        seg_btn_row.setSpacing(8)
        self.seg_start_btn = QPushButton(tr("btn_seg_start"))
        self.seg_end_btn = QPushButton(tr("btn_seg_end"))
        self.seg_start_btn.setEnabled(False)
        self.seg_end_btn.setEnabled(False)
        self.seg_start_btn.clicked.connect(self._seg_start)
        self.seg_end_btn.clicked.connect(self._seg_stop)
        seg_btn_row.addWidget(self.seg_start_btn)
        seg_btn_row.addWidget(self.seg_end_btn)
        seg_btn_row.addStretch(1)
        seg_form.addRow("", seg_btn_row)
        self.seg_hint = QLabel(tr("seg_hint_idle"))
        self._seg_hint_key = "seg_hint_idle"
        self.seg_hint.setObjectName("panelSub")
        self.seg_hint.setWordWrap(True)
        seg_form.addRow("", self.seg_hint)
        left.addWidget(self._seg)

        left.addStretch(1)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_widget)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setMinimumWidth(380)

        # 底部状态行（标定/出流进度、错误提示）：放在滚动区**之外**，窗口再矮也始终可见
        self.status = QLabel(tr("status_ready"))
        self.status.setObjectName("panelSub")
        self.status.setWordWrap(True)
        left_pane = QWidget()
        left_pane_layout = QVBoxLayout(left_pane)
        left_pane_layout.setContentsMargins(0, 0, 0, 0)
        left_pane_layout.setSpacing(8)
        left_pane_layout.addWidget(left_scroll, 1)
        left_pane_layout.addWidget(self.status)

        # ---- 右侧：实时预览（5 行手指 × 2 列左右手；点击单格放大，再点还原） ----
        self._tiles = {}          # SN -> SensorTile
        self._tile_order = []     # 建格顺序，网格按此稳定排序
        self._tile_labels = {}    # SN -> '{hand}_{finger}'（决定格子在网格里的行列）
        self._zoomed = None       # 当前放大的 SN（None = 网格视图）
        self._placeholder = QLabel(tr("preview_no_stream"))
        self._placeholder.setObjectName("preview")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_widget = QWidget()
        right_widget.setObjectName("panel")
        self._tile_grid = QGridLayout(right_widget)
        self._tile_grid.setContentsMargins(0, 0, 0, 0)
        self._tile_grid.setSpacing(6)
        self._tile_grid.addWidget(self._placeholder, 0, 0)

        # 左右两栏用 QSplitter：初始各占一半，可拖动调整
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        splitter.addWidget(left_pane)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 500])
        outer.addWidget(splitter, 1)

        self._refresh_path()
        self.setFocus()
        _RETRANSLATE.append(self.retranslate)

    def _refresh_path(self):
        task = _sanitize_name(self.task_edit.text(), "unnamed")
        self.path_label.setText(os.path.join(DATA_DIR, task) + "/")

    # -- 语言切换 --
    def _toggle_lang(self):
        set_language("en" if LANG == "zh" else "zh")

    def retranslate(self):
        """语言切换时刷新本面板所有静态文案；动态状态由各自槽函数在下次触发时用 tr() 重取。"""
        self._title.setText(tr("app_title"))
        self._sub.setText(tr("panel_subtitle"))
        self.back_viewer_btn.setText(tr("btn_back_viewer"))
        self.lang_btn.setText(tr("lang_switch"))
        self.detect_btn.setText(tr("btn_detect"))
        self.import_btn.setText(tr("btn_import"))
        self.start_btn.setText(tr("btn_start"))
        self.stop_btn.setText(tr("btn_stop"))
        self._store.setTitle(tr("store_title"))
        self.task_edit.setPlaceholderText(tr("task_placeholder"))
        self._store_form.labelForField(self.task_edit).setText(tr("label_folder"))
        self._store_form.labelForField(self.path_label).setText(tr("label_path"))
        self._save_box.setTitle(tr("save_group"))
        for key, cb in self._save_checks.items():
            cb.setText(tr(f"save_{key}"))
        self.recalib_btn.setText(tr("btn_recalib"))
        self._seg.setTitle(tr("seg_title"))
        self.seg_start_btn.setText(tr("btn_seg_start"))
        self.seg_end_btn.setText(tr("btn_seg_end"))
        self.seg_hint.setText(tr(self._seg_hint_key))
        if not self._tiles:
            self._placeholder.setText(tr("preview_no_stream"))
        if self._worker is None:
            self.status.setText(tr("status_ready"))

    # -- 动作 --
    def _detect(self):
        if self._detector is not None and self._detector.isRunning():
            return
        self.detect_btn.setEnabled(False)
        self.status.setText(tr("detecting"))
        self._detector = DetectWorker()
        self._detector.devices_found.connect(self._on_devices_found)
        self._detector.status.connect(self._on_status)
        self._detector.finished.connect(self._on_detect_finished)
        self._detector.start()

    def _import_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("import_dialog_title"), CONFIG_DIR, tr("import_filter"))
        if not path:
            return
        ok, msg = self._finger_widget.import_config(path)
        if ok:
            self.status.setText(msg)
        else:
            QMessageBox.warning(self, tr("import_warn_title"), msg)

    def _start(self):
        # 防重入：整个启动流程（含下面的基线确认弹窗）期间只允许一次。
        # 否则弹窗还开着时又来了点击/回车，会叠出第二个确认框并起第二个采集线程。
        if self._starting or (self._worker is not None and self._worker.isRunning()):
            return
        self._starting = True
        try:
            sensors_sn, labels = self._finger_widget.collect_assign()
            if not sensors_sn:
                QMessageBox.warning(self, tr("start_no_finger_title"), tr("start_no_finger_msg"))
                return
            if len(set(sensors_sn)) != len(sensors_sn):
                QMessageBox.warning(self, tr("start_dup_title"), tr("start_dup_msg"))
                return
            # 基线保护：标定与首帧基线要求传感器处于无按压状态，开始前显式确认
            ans = QMessageBox.question(self, tr("baseline_confirm_title"),
                                       tr("baseline_confirm_msg"))
            if ans != QMessageBox.StandardButton.Yes:
                return
            self._finger_widget.save_config()
            self._worker = DataCaptureWorker(sensors_sn, labels)
            self._worker.status.connect(self._on_status)
            self._worker.frame.connect(self._on_frame)
            self._worker.streaming.connect(self._on_streaming)
            self._worker.episode_changed.connect(self._on_episode)
            self._worker.finished.connect(self._on_finished)
            self._push_spec()  # 注入当前渲染模式，出流第一帧就按正确模式跑
            self._worker.start()
            self.detect_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self._finger_widget.set_enabled(False)
            self.stop_btn.setEnabled(True)
            self.status.setText(tr("starting"))
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
        return self._worker is not None and self._worker.isRunning() and self._worker.episode_active()

    def discard_recording(self):
        """请求丢弃当前正在录制的 episode（停止采集时直接删除，不落盘）。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.discard_episode()

    def shutdown(self):
        """关闭窗口/退出时调用：停 detector/worker 线程并等待其释放传感器。"""
        if self._detector is not None:
            if self._detector.isRunning():
                self._detector.wait(1000)
            self._detector = None
        if self._worker is not None:
            self._worker.request_stop()
            if self._worker.isRunning():
                if not self._worker.wait(5000):
                    self.status.setText(tr("shutdown_timeout"))
        self._worker = None

    # -- episode 录制动作 --
    def _seg_start(self):
        if self._worker is None or not self._worker.isRunning():
            return
        task = _sanitize_name(self.task_edit.text(), "")
        if not task:
            QMessageBox.warning(self, tr("seg_no_task_title"), tr("seg_no_task_msg"))
            return
        if not any(cb.isChecked() for cb in self._save_checks.values()):
            QMessageBox.warning(self, tr("start_no_type_title"), tr("start_no_type_msg"))
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
            "task": _sanitize_name(self.task_edit.text(), ""),
            "save": save,
        })

    def _refresh_recording_ui(self):
        streaming = self._worker is not None and self._worker.isRunning()
        recording = streaming and self._worker.episode_active()
        self.seg_start_btn.setEnabled(streaming and not recording)
        self.seg_end_btn.setEnabled(recording)
        if not streaming:
            self._seg_hint_key = "hint_no_stream"
        elif not recording:
            self._seg_hint_key = "hint_ready"
        else:
            self._seg_hint_key = "hint_recording"
        self.seg_hint.setText(tr(self._seg_hint_key))

    # -- 信号槽 --
    def _on_status(self, msg):
        self.status.setText(msg)

    def _on_devices_found(self, sns):
        self._sns = list(sns)
        self._finger_widget.set_sns(sns)
        self.start_btn.setEnabled(True)

    def _on_detect_finished(self):
        if self._detector is not None:
            self._detector.wait(1000)
            self._detector = None
        self.detect_btn.setEnabled(True)

    def _on_streaming(self, on):
        self.recalib_btn.setEnabled(bool(on))
        if not on:
            self._clear_tiles()
        self._refresh_recording_ui()

    def _on_episode(self, on, desc):
        # 结束时状态栏保留 worker 刚发出的 episode_ended_detail（帧数/体积/丢帧）
        if on:
            self.status.setText(tr("episode_recording", desc=desc))
        self._refresh_recording_ui()

    def _on_frame(self, payload):
        """采集线程隔帧推来的预览包 {"tiles", "stats", "labels"}；新传感器首次出现时建格。"""
        try:
            tiles = payload.get("tiles") or {}
            stats = payload.get("stats") or {}
            labels = payload.get("labels") or {}
            for sn, bgr in tiles.items():
                lbl = labels.get(sn) or ""
                self._tile_labels[sn] = lbl
                tile = self._tiles.get(sn)
                if tile is None:
                    tile = SensorTile(sn)
                    tile.clicked.connect(self._on_tile_clicked)
                    self._tiles[sn] = tile
                    self._tile_order.append(sn)
                    self._relayout_tiles()   # 标签已就位，新格直接落到正确的行列
                tile.set_content(bgr, f"{lbl} · {sn}" if lbl else sn, stats.get(sn))
        except Exception:  # noqa: BLE001
            pass

    def _tile_cell(self, sn):
        """SN -> (行, 列)：行 = 手指序（拇指→小指），列 = 0 左手 / 1 右手。

        没有手指标定（或标签不在 HANDS×FINGERS 内）时返回 None。
        """
        hand, _, finger = (self._tile_labels.get(sn) or "").partition("_")
        if hand in HANDS and finger in FINGERS:
            return FINGERS.index(finger), (0 if hand == "left" else 1)
        return None

    def _relayout_tiles(self):
        """重排卡片网格：5 行(手指) × 2 列(左手在左、右手在右)；_zoomed 指向某格时该格单独铺满。

        有任何一格没有手指标定时，退回「按建格顺序每行 2 列」，未标定的传感器不会被丢掉。
        只标定部分手指时，空缺的行/列会被 QGridLayout 收缩。

        布局操作有两处硬约束，改这段时别丢：
          1) takeAt() 之后只 hide()，**不要 setParent(None)**——解除父级会让 widget 变成
             真正的顶层窗口，桌面上就会一台传感器冒出一个独立小窗口（还会自己消失）；
          2) 重新入格必须「先 addWidget 认父、再 show()」，否则同样会闪出一个独立窗口。
        """
        while self._tile_grid.count():
            item = self._tile_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()   # 只移出布局 + 隐藏；父级保持为右侧容器，就不会变成窗口
        sns = [sn for sn in self._tile_order if sn in self._tiles]
        if self._zoomed in self._tiles:
            for sn, t in self._tiles.items():
                if sn != self._zoomed:
                    t.hide()
            zoomed = self._tiles[self._zoomed]
            self._tile_grid.addWidget(zoomed, 0, 0)
            zoomed.show()
            self._placeholder.hide()
            return
        self._zoomed = None
        if not sns:
            self._tile_grid.addWidget(self._placeholder, 0, 0)
            self._placeholder.show()
            return
        self._placeholder.hide()
        cells = [self._tile_cell(sn) for sn in sns]
        finger_grid = all(c is not None for c in cells)
        for i, sn in enumerate(sns):
            t = self._tiles[sn]
            r, c = cells[i] if finger_grid else (i // 2, i % 2)
            self._tile_grid.addWidget(t, r, c)
            t.show()

    def _on_tile_clicked(self, sn):
        self._zoomed = None if self._zoomed == sn else sn
        self._relayout_tiles()

    def _clear_tiles(self):
        for t in self._tiles.values():
            t.hide()          # 父级保持不动（见 _relayout_tiles 的注释），hide 后交给 deleteLater
            t.deleteLater()
        self._tiles = {}
        self._tile_order = []
        self._tile_labels = {}
        self._zoomed = None
        self._relayout_tiles()

    def _recalibrate(self):
        """「重新标定」：SDK 重设背景，深度/marker 基线随下一帧重建。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_recalibrate()
        self.setFocus()

    def _on_finished(self):
        if self._worker is not None:
            self._worker.wait(5000)
            self._worker = None
        self.detect_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self._finger_widget.set_enabled(True)
        self.seg_start_btn.setEnabled(False)
        self.seg_end_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def keyPressEvent(self, event):
        k = event.key()
        if k in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self._shutdown()
            event.accept()
            return
        super().keyPressEvent(event)


# --------------------------------------------------------------------------- #
# 主题
# --------------------------------------------------------------------------- #
LIGHT_QSS = """
* { font-family: -apple-system, "SF Pro Text", "Helvetica Neue", "PingFang SC",
    "Microsoft YaHei", sans-serif; }
QMainWindow, QWidget#panel { background: #FFFFFF; }
QLabel { color: #1D1D1F; }
QLabel#panelTitle { font-size: 22px; font-weight: 700; color: #1D1D1F; }
QLabel#panelSub { font-size: 13px; color: #8E8E93; }
QLabel#preview { background: #F5F5F7; border: 1px solid #8E8E93; border-radius: 10px; color: #8E8E93; }
QGroupBox { font-weight: 600; color: #1D1D1F; border: 1px solid #D1D1D6; border-radius: 10px;
    margin-top: 10px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QLineEdit, QComboBox {
    background: #F5F5F7; border: 1px solid #8E8E93; border-radius: 10px;
    padding: 7px 12px; color: #1D1D1F; min-height: 22px;
}
QLineEdit:focus, QComboBox:focus { border: 1px solid #3A3A3C; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: #FFFFFF; border: 1px solid #8E8E93; border-radius: 10px;
    selection-background-color: #E3E3E8; }
QPushButton { background: #F2F2F5; border: 1px solid #8E8E93; border-radius: 10px;
    padding: 6px 16px; color: #1D1D1F; }
QPushButton:hover { background: #E8E8ED; }
QPushButton:disabled { color: #B0B0B5; background: #F2F2F5; border-color: #D1D1D6; }
QCheckBox { color: #1D1D1F; spacing: 8px; }
QScrollArea { border: none; background: #FFFFFF; }
QScrollArea > QWidget > QWidget { background: #FFFFFF; }
"""

DARK_QSS = """
* { font-family: -apple-system, "SF Pro Text", "Helvetica Neue", "PingFang SC",
    "Microsoft YaHei", sans-serif; }
QMainWindow, QWidget#panel { background: #1C1C1E; }
QLabel { color: #F5F5F7; }
QLabel#panelTitle { font-size: 22px; font-weight: 700; color: #F5F5F7; }
QLabel#panelSub { font-size: 13px; color: #98989D; }
QLabel#preview { background: #2C2C2E; border: 1px solid #8E8E93; border-radius: 10px; color: #98989D; }
QGroupBox { font-weight: 600; color: #F5F5F7; border: 1px solid #3A3A3C; border-radius: 10px;
    margin-top: 10px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QLineEdit, QComboBox {
    background: #2C2C2E; border: 1px solid #8E8E93; border-radius: 10px;
    padding: 7px 12px; color: #F5F5F7; min-height: 22px;
}
QLineEdit:focus, QComboBox:focus { border: 1px solid #D1D1D6; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: #2C2C2E; border: 1px solid #8E8E93; border-radius: 10px;
    selection-background-color: #3A3A3C; }
QPushButton { background: #3A3A3C; border: 1px solid #8E8E93; border-radius: 10px;
    padding: 6px 16px; color: #F5F5F7; }
QPushButton:hover { background: #48484A; }
QPushButton:disabled { color: #636366; background: #2C2C2E; border-color: #3A3A3C; }
QCheckBox { color: #F5F5F7; spacing: 8px; }
QScrollArea { border: none; background: #1C1C1E; }
QScrollArea > QWidget > QWidget { background: #1C1C1E; }
"""


# --------------------------------------------------------------------------- #
# 自绘复选框（参考 widgets/checkable_combo.py 的 _CheckColorDelegate）
# 直接 override paintEvent 用 QPainter 画勾选框与文字，完全不依赖原生指示器
# 或 QSS 的 ::indicator，保证在各平台/样式下边框与对号都稳定显示。
# --------------------------------------------------------------------------- #
# 深色主题标志：None=自动感知（供 main.py 宿主等未调用 _set_check_dark 的入口），
# True/False=强制指定（vitai_capture_gui 自己的 main()/深色菜单开关）。
_CHECK_DARK = None


def _set_check_dark(on):
    global _CHECK_DARK
    _CHECK_DARK = on


class ColorCheckBox(QCheckBox):
    """自绘勾选框：文字与勾选框用同一支画笔绘制，颜色永远一致并随主题变化，
    不填充背景。"""

    _BOX = 18
    _GAP = 8

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        # 关掉原生 spacing，自绘时自行控制文字与勾选框间距
        self.setStyleSheet("QCheckBox { spacing: 0px; }")

    def _text_color(self):
        """取与当前主题文字一致的颜色。

        - 宿主入口（main.py）只设置调色板、不会调用 _set_check_dark，
          此时直接用调色板的 WindowText（即主题定义的文字颜色）。
        - 本模块自己的 main()/深色开关会强制 _CHECK_DARK，用固定常量，
          与 QSS 中 QCheckBox 的 color 保持一致。
        """
        if not self.isEnabled():
            return QColor("#9A9A9A")
        if _CHECK_DARK is None:
            c = self.palette().color(QPalette.ColorRole.WindowText)
            if c.isValid() and c.lightness() > 0:
                return c
            dark = self.palette().color(
                QPalette.ColorRole.Window).lightness() < 128
            return QColor("#F5F5F7") if dark else QColor("#1D1D1F")
        return QColor("#F5F5F7") if _CHECK_DARK else QColor("#1D1D1F")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        color = self._text_color()

        # 自绘勾选框：描边对齐像素网格（0.5 偏移 + 1px 线宽），
        # 避免抗锯齿把线条冲淡成灰色，保证颜色与文字一致。
        box = QRect(0, (self.height() - self._BOX) // 2, self._BOX, self._BOX)
        box_f = QRectF(box.x() + 0.5, box.y() + 0.5,
                       box.width() - 1, box.height() - 1)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, 1.0))
        painter.drawRoundedRect(box_f, 3.5, 3.5)
        if self.isChecked():
            painter.setPen(QPen(color, 2.2, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            x, y, w, h = box_f.x(), box_f.y(), box_f.width(), box_f.height()
            painter.drawLine(QPointF(x + w * 0.24, y + h * 0.52),
                             QPointF(x + w * 0.42, y + h * 0.70))
            painter.drawLine(QPointF(x + w * 0.42, y + h * 0.70),
                             QPointF(x + w * 0.76, y + h * 0.30))

        # 自绘文字：与勾选框使用完全相同的画笔颜色，二者永远一致
        text_rect = self.rect().adjusted(self._BOX + self._GAP, 0, 0, 0)
        painter.setFont(self.font())
        painter.setPen(QPen(color, 1.0))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.text(),
        )
        painter.end()


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self, dark=False):
        super().__init__()
        self.setWindowTitle(tr("app_title"))
        # 窗口自适应当前屏幕：取可用屏幕尺寸的 90%
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.9), int(screen.height() * 0.9))
        self.setMinimumSize(760, 560)

        # 视图菜单：深色模式
        self._menu = self.menuBar().addMenu(tr("menu_view"))
        self.dark_act = self._menu.addAction(tr("menu_dark"))
        self.dark_act.setCheckable(True)
        self.dark_act.setChecked(dark)
        self.dark_act.toggled.connect(
            lambda on: (
                _set_check_dark(on),
                QApplication.instance().setStyleSheet(DARK_QSS if on else LIGHT_QSS),
            )
        )

        self.panel = CapturePanel()
        self.setCentralWidget(self.panel)
        _RETRANSLATE.append(self.retranslate)

    def retranslate(self):
        self.setWindowTitle(tr("app_title"))
        self._menu.setTitle(tr("menu_view"))
        self.dark_act.setText(tr("menu_dark"))

    def closeEvent(self, event):
        self.panel.shutdown()
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main():
    app = QApplication(sys.argv)
    # 复选框用自绘 ColorCheckBox（见文件上方），不依赖原生指示器或 QSS ::indicator，
    # 保证边框/对号在各平台/样式下都能稳定渲染。
    dark = "--dark" in sys.argv
    _set_check_dark(dark)
    app.setApplicationName(tr("app_title"))
    app.setStyleSheet(DARK_QSS if dark else LIGHT_QSS)
    # 让默认控件背景跟随主题（主内容区纯白 / 深色），避免视口/普通 widget 显示系统灰
    pal = app.palette()
    bg = QColor("#1C1C1E") if dark else QColor("#FFFFFF")
    pal.setColor(QPalette.ColorRole.Window, bg)
    pal.setColor(QPalette.ColorRole.Base, bg)
    app.setPalette(pal)
    win = MainWindow(dark=dark)
    win.show()
    if "--smoke" in sys.argv:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, app.quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
