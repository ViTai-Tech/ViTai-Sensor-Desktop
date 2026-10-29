# -*- coding: utf-8 -*-
"""标记点云 + 位移箭头控件（纯 Qt 绘制，正交 3D 投影）。

数据来自 SDK 的 ``MARKER_ORIGIN_VECTOR`` / ``MARKER_CURRENT_VECTOR``：

- 形状兼容 ``(H, W, 3)``（如 20×20×3、20×10×3）、``(N, 3)``、``(N, 2)``
  以及对应的一维扁平数组；缺 z 时按 0 处理；
- 点云画在「当前点」上，颜色由该点的位移长度决定；
- 位移超过阈值的点不再画静止圆点，而是画一条「原始点 → 当前点」的 3D 箭头
  （与原 JS 的 PointCloudRenderer 逻辑一致：有箭头时隐藏当前粒子）；
- 视图采用「正交旋转 + 透视除法」的相机模型：缩放系数只按「默认朝向」的投影
  包围盒计算并固定下来，因此无论数据是 20×20×3 还是 20×10×3 都能铺满外框；
  透视相机到点云中心的距离随点阵真实尺度固定，旋转时不改变整体取景。翻转时
  近处的点更大、间距更开，远处的点更小、更收拢，整体呈梯形（近大远小），
  而不是所有间隙一起等比缩小。

相机：默认方位角 0°、仰角 25°（相机在点云平面正前方略微抬高），绕点云中心
做完整的 3D 轨道旋转：左右拖拽 = 绕 y 轴（屏幕竖直轴）偏航，点云平面会像转盘
一样侧过去；上下拖拽 = 绕 x 轴（屏幕水平轴）俯仰。默认朝向下点云平面在屏幕上
保持正放的方形，z 是垂直平面的深度方向，在屏幕上表现为竖直偏移，因此箭头能
直观看到「原点的 z」指向「当前点的 z」。

交互（与 ``depth_3d_view`` 保持一致）：左键拖拽旋转、右键拖拽平移、滚轮缩放、
双击复位；按下时切换为「抓手」光标，松开恢复。渲染用 QPainter 完成，
点数量很小（几百个），无需占用 OpenGL 上下文，也不会和深度 3D 控件抢 GL。
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget


_BG = QColor("#070b14")
_BORDER = QColor("#1a3a5c")
_MUTED = QColor("#475569")
_LOADING = QColor("#67e8f9")

# 位移色带：青（位移小）-> 绿 -> 黄 -> 红（位移大）
_COLOR_STOPS = (
    (0.00, (0.13, 0.83, 0.93)),
    (0.35, (0.20, 0.90, 0.45)),
    (0.65, (0.98, 0.90, 0.36)),
    (1.00, (0.97, 0.45, 0.32)),
)

# 默认相机：方位角 0°、仰角 25°。
# 方位角是绕 y 轴（屏幕竖直轴）的偏航，0° 时相当于正对点云平面；仰角是绕 x 轴
# （屏幕水平轴）的俯仰，25° 让相机从平面正前方略微抬高。默认朝向下 x-y 网格
# 在屏幕上仍是正放的方形（不会变成菱形），z 在屏幕上体现为竖直方向的偏移，
# 箭头才能看出去「原点的 z」到「当前点的 z」。
_DEFAULT_AZIMUTH = 0.0
_DEFAULT_ELEVATION = math.radians(25.0)

# 画布留白（像素），保证最外侧的点 / 箭头不会被边框裁掉。
_MARGIN = 16.0

# 位移阈值（占点云包围盒对角线的比例）：低于该值视为静止点，只画圆点。
_MOVE_THRESHOLD_RATIO = 0.004

# z 坐标放大系数：原始 z 数值通常很小，按真实值显示时高度几乎看不见。
# 这里只在「显示投影」阶段把 z 放大该倍数（竖杆高度、屏幕偏移都更明显），
# 不参与位移长度 / 颜色的计算，因此不会改变位移着色。
# 取正值：z 增大时点向上抬（深度轴朝上）；透视深度 w 也用真实 z，方向与显示一致。
# 需要把深度轴反过来时，把它改成负值即可。
_Z_SCALE = 500.0

# 透视相机距离系数：相机沿视线方向到点云中心的距离 = 场景半径 × 该系数。
# 系数越小透视越强（近大远小 / 翻转时的梯形越明显），越大越接近正交投影。
# 取值需 > 1 才能保证所有点都落在相机前方（分母 cam_dist - w 恒为正）。
_PERSPECTIVE_DISTANCE = 2.2

# 红色整体位移箭头的可见阈值：位移幅值（各分量的欧氏范数）小于该值时不绘制。
# 先用较小的数测试，确认效果后再调整。
_DELTA_ARROW_VISIBLE_THRESHOLD = 0.05


def _lerp_color(t: float, alpha: int = 255) -> QColor:
    """把 0..1 的归一化位移映射为色带上的颜色。"""
    t = min(max(t, 0.0), 1.0)
    for i in range(len(_COLOR_STOPS) - 1):
        t0, c0 = _COLOR_STOPS[i]
        t1, c1 = _COLOR_STOPS[i + 1]
        if t <= t1:
            f: float = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            r = c0[0] + (c1[0] - c0[0]) * f
            g = c0[1] + (c1[1] - c0[1]) * f
            b = c0[2] + (c1[2] - c0[2]) * f
            return QColor(int(r * 255), int(g * 255), int(b * 255), alpha)
    _, c_last = _COLOR_STOPS[-1]
    return QColor(
        int(c_last[0] * 255), int(c_last[1] * 255), int(c_last[2] * 255), alpha
    )


def _to_nx3(value) -> np.ndarray | None:
    """把各种可能的输入形状统一成 (N, 3) 的浮点数组，失败时返回 None。"""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return None

    if arr.ndim == 1:
        if arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        elif arr.size % 2 == 0:
            arr = np.column_stack([arr.reshape(-1, 2), np.zeros(arr.size // 2)])
        else:
            return None
    elif arr.ndim == 2:
        if arr.shape[1] >= 3:
            arr = arr[:, :3]
        elif arr.shape[1] == 2:
            arr = np.column_stack([arr, np.zeros(arr.shape[0])])
        elif arr.shape[0] in (2, 3):  # (3, N) 这种转置布局
            arr = arr.T
            if arr.shape[1] == 2:
                arr = np.column_stack([arr, np.zeros(arr.shape[0])])
            else:
                arr = arr[:, :3]
        else:
            return None
    elif arr.ndim == 3:
        # (H, W, C) 或 (C, H, W)
        if arr.shape[-1] in (2, 3):
            pass
        elif arr.shape[0] in (2, 3):
            arr = np.moveaxis(arr, 0, -1)
        else:
            return None
        c = arr.shape[-1]
        arr = arr.reshape(-1, c)
        if c == 2:
            arr = np.column_stack([arr, np.zeros(arr.shape[0])])
        else:
            arr = arr[:, :3]
    else:
        return None

    arr = arr[np.isfinite(arr).all(axis=1)]
    return arr if arr.shape[0] else None


class PointCloudWidget(QWidget):
    """点云 + 位移箭头显示控件（接口与 ImageLabel / Depth3DWidget 对齐）。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(120, 90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self._origin: np.ndarray | None = None
        self._current: np.ndarray | None = None
        self._delta_xyz: np.ndarray | None = None
        self._loading = False

        self._azimuth = _DEFAULT_AZIMUTH
        self._elevation = _DEFAULT_ELEVATION
        self._zoom = 1.0
        # 右键平移：屏幕像素偏移，直接叠加到投影中心上（1.0 = 自适应满屏）。
        self._pan_x = 0.0
        self._pan_y = 0.0
        # 是否在每个当前点旁标注其 z 坐标数值（默认开；点很多时只标发生位移的点）。
        self._show_z_labels = True

        self._button = Qt.MouseButton.NoButton
        self._last_pos: QPointF | None = None

    # ------------------------------------------------------------ 尺寸策略
    def sizeHint(self) -> QSize:
        return QSize(160, 120)

    def minimumSizeHint(self) -> QSize:
        return QSize(120, 90)

    # ------------------------------------------------------------ public API
    def set_points(self, origin, current, delta_xyz) -> None:
        """每收到一帧 marker 数据调用一次。

        ``origin`` = MARKER_ORIGIN_VECTOR，``current`` = MARKER_CURRENT_VECTOR。
        两个数组按行一一对应；长度不一致时截断到较短的一方。任一为 None 时
        退化为只画点云（origin 与 current 相同）。
        """
        cur = _to_nx3(current)
        org = _to_nx3(origin)

        if cur is None and org is None:
            self.clear()
            return
        if cur is None:
            cur = org
        if org is None:
            org = cur
        if cur is None or org is None:
            return

        n = min(org.shape[0], cur.shape[0])
        # 强制拷贝：SDK 可能复用同一块缓冲区，避免绘制前被下一帧改写。
        self._origin = np.array(org[:n], copy=True)
        self._current = np.array(cur[:n], copy=True)
        dlt = _to_nx3(delta_xyz)
        if dlt is not None and dlt.shape[0] > 0:
            self._delta_xyz = np.array(dlt[: min(dlt.shape[0], n)], copy=True)
        else:
            self._delta_xyz = None
        self._loading = False
        self.update()

    # 兼容旧的单数调用名
    set_point = set_points

    def clear(self) -> None:
        self._origin = None
        self._current = None
        self._delta_xyz = None
        self._loading = False
        self.update()

    def set_loading(self, loading: bool) -> None:
        self._loading = bool(loading)
        if loading:
            self._origin = None
            self._current = None
            self._delta_xyz = None
        self.update()

    def set_show_z_labels(self, show: bool) -> None:
        """是否在每个当前点旁标出其 z 坐标数值（默认开）。"""
        self._show_z_labels = bool(show)
        self.update()

    def release_resources(self) -> None:
        # 纯 QPainter 控件，无需释放 GL 资源；保留接口以便统一调用。
        return

    def reset_view(self) -> None:
        self._azimuth = _DEFAULT_AZIMUTH
        self._elevation = _DEFAULT_ELEVATION
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    # ------------------------------------------------------------ 投影
    @staticmethod
    def _project_at(pts: np.ndarray, azimuth: float, elevation: float):
        """按指定偏航 / 俯仰角做「正交旋转」：返回 u（右）、v（下）、w（深度，越大越近）。

        相机围绕点云中心做两段旋转（先绕 y 轴偏航、再绕 x 轴俯仰），因此左右 /
        上下拖拽都是真正的 3D 旋转：

        - 偏航 ``azimuth``：绕世界 y 轴（屏幕竖直轴）旋转，x 与 z 相互混合。
          点云平面会像转盘一样侧过去，z 方向的位移也会获得水平分量，所以箭头
          能看出「绕竖直轴转」的立体感；
        - 俯仰 ``elevation``：绕世界 x 轴（屏幕水平轴）旋转，y 与 z 相互混合，
          表现为屏幕竖直方向的偏移：z 越大越靠上。

        这里只做旋转，不做透视除法；近大远小由 ``_project_perspective`` 在旋转
        结果上再做一次透视缩放实现。``azimuth = 0``、``elevation = 0`` 时是正对
        x-y 平面的正视（x 向右、y 向下、z 指向观察者）；默认仰角给 25°，z 差值
        能在竖直方向显示出来。
        """
        ca, sa = math.cos(azimuth), math.sin(azimuth)
        ce, se = math.cos(elevation), math.sin(elevation)

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        # 1) 绕 y 轴偏航（左右拖拽）：在 x-z 平面内旋转，点云平面像转盘一样侧过去，
        #    z 位移因此也会产生水平分量 —— 这才是绕竖直轴的 3D 旋转，而不是把
        #    网格当成一张图片在屏幕里转圈（那样 z 永远只沿竖直方向，看不出 3D）。
        x1 = x * ca + z * sa
        z1 = z * ca - x * sa

        # 2) 绕 x 轴俯仰（上下拖拽）：在 y-z 平面内旋转（与原来的仰角一致）
        u = x1
        v = y * ce - z1 * se
        w = y * se + z1 * ce
        return u, v, w

    @staticmethod
    def _project_perspective(
        disp_pts: np.ndarray,
        depth_pts: np.ndarray,
        azimuth: float,
        elevation: float,
        cam_dist: float,
    ):
        """正交旋转 + 透视除法，返回屏幕坐标 u/v、深度 w 与透视因子 f。

        - 屏幕坐标 ``u``/``v`` 用 ``disp_pts``（z 已按 ``_Z_SCALE`` 放大）旋转得到，
          决定点实际画在哪里、平面看起来多宽;
        - 深度 ``w`` 用 ``depth_pts``（真实几何，z 未放大）旋转得到，只用于透视除法
          与前后排序，避免 z 的显示放大把透视比例带偏;
        - 透视因子 ``f = cam_dist / (cam_dist - w)``：w 越大（越靠近相机）f 越大，
          该点被放大；越远 f 越小。用它乘 u/v 即得近大远小的梯形效果。

        ``cam_dist`` 需大于所有点的 ``|w|``，由调用方按场景半径 × 系数保证，
        这里再用下限兜底，避免除零 / 负深度。
        """
        u, v, _ = PointCloudWidget._project_at(disp_pts, azimuth, elevation)
        _, _, w = PointCloudWidget._project_at(depth_pts, azimuth, elevation)
        denom = np.maximum(cam_dist - w, cam_dist * 1e-3)
        f = cam_dist / denom
        return u * f, v * f, w, f

    def _fit_scale(self, rect: QRectF, u_all: np.ndarray, v_all: np.ndarray) -> float:
        """按给定投影包围盒算出「刚好铺满控件」的缩放系数。

        注意：这里只用**固定参考朝向**的包围盒来算，不能每帧用当前朝向算。
        否则旋转时投影宽度变小、缩放系数自动变大，整团点云会被越拉越大
        （看起来像被拉长、整体取景跳动）。近大远小的透视形变由 ``w`` 的透视
        除法负责，与这里的取景缩放相互独立。
        """
        u_min, u_max = float(u_all.min()), float(u_all.max())
        v_min, v_max = float(v_all.min()), float(v_all.max())
        span_u = max(u_max - u_min, 1e-9)
        span_v = max(v_max - v_min, 1e-9)

        avail_w = max(1.0, rect.width() - 2.0 * _MARGIN)
        avail_h = max(1.0, rect.height() - 2.0 * _MARGIN)
        return min(avail_w / span_u, avail_h / span_v) * self._zoom

    # ------------------------------------------------------------ 绘制
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        p.setPen(QPen(_BORDER, 1.0))
        p.setBrush(_BG)
        p.drawRoundedRect(rect, 8.0, 8.0)

        if self._loading:
            p.setPen(_LOADING)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "加载中...")
            p.end()
            return

        if self._current is None or self._origin is None or self._current.size == 0:
            p.setPen(_MUTED)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "无信号")
            p.end()
            return

        org = self._origin
        cur = self._current

        # 投影 / 旋转中心固定在「参考构型（原始点 origin）」的包围盒中心，
        # 而不是原始点 + 当前点共同包围盒的中心。
        # 这样旋转时绕的是参考构型的中心：初始面（origin 点）以自身中心为支点做
        # 纯 3D 旋转，位置稳定、不会随当前点的位移而漂移；即使某个当前点的 z 很大，
        # 静止时初始面也始终不动，当前点只是相对它发生位移，符合「绕原点旋转」的直觉。
        center = 0.5 * (org.min(axis=0) + org.max(axis=0))
        # 真实几何（未放大 z）：只用于推导透视深度 / 相机距离，让近大远小的比例
        # 只取决于点阵本身的真实尺度，不被 z 的显示放大污染。
        base_o = org - center
        base_c = cur - center
        # 显示坐标：把 z 放大，让竖杆和高度差在屏幕上看得见（只影响屏幕位置）。
        disp_o = base_o.copy()
        disp_c = base_c.copy()
        disp_o[:, 2] *= _Z_SCALE
        disp_c[:, 2] *= _Z_SCALE

        # 透视相机：相机到点云中心的距离 = 场景半径 × 固定系数。半径取自真实几何的
        # 包围球，与旋转角无关，所以旋转时透视强度恒定，不会「转一下就整体胀缩」。
        radius = float(
            max(
                float(np.linalg.norm(base_o, axis=1).max()),
                float(np.linalg.norm(base_c, axis=1).max()),
                1e-9,
            )
        )
        cam_dist = _PERSPECTIVE_DISTANCE * radius

        # 当前视角：先做正交旋转得到 u/v（显示坐标）与 w（真实深度），再做透视除法。
        # w 越大越靠近相机 → 透视因子 cam_dist/(cam_dist-w) 越大 → 该点被放大、
        # 与邻点的间距也更大；远处则被压缩。翻转平面时近端撑开、远端收拢，整体呈
        # 梯形（近大远小），而不是所有间隙一起等比缩小。
        u_o, v_o, _w_o, f_o = self._project_perspective(
            disp_o, base_o, self._azimuth, self._elevation, cam_dist
        )
        u_c, v_c, w_c, f_c = self._project_perspective(
            disp_c, base_c, self._azimuth, self._elevation, cam_dist
        )

        # 缩放系数固定按「默认朝向」计算（同样带透视），与当前旋转角无关：这样转的
        # 时候整体取景不跳变，只有近大远小的透视形变在变化。
        ru_o, rv_o, _, _ = self._project_perspective(
            disp_o, base_o, 0.0, _DEFAULT_ELEVATION, cam_dist
        )
        ru_c, rv_c, _, _ = self._project_perspective(
            disp_c, base_c, 0.0, _DEFAULT_ELEVATION, cam_dist
        )
        scale = self._fit_scale(
            rect, np.concatenate([ru_o, ru_c]), np.concatenate([rv_o, rv_c])
        )

        # 投影前已减去 3D 包围盒中心，投影面中点即为 0；右键平移按屏幕像素叠加。
        cx = rect.center().x() + self._pan_x
        cy = rect.center().y() + self._pan_y

        def sx(u):
            return cx + u * scale

        def sy(v):
            return cy + v * scale

        # 位移与颜色归一化
        disp = np.linalg.norm(cur - org, axis=1)
        all_pts = np.vstack([org, cur])
        diag = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)))
        if diag < 1e-9:
            diag = 1.0
        move_thr = diag * _MOVE_THRESHOLD_RATIO

        positive = disp[disp > move_thr]
        ref = float(np.percentile(positive, 95)) if positive.size else 1.0
        ref = max(ref, move_thr, 1e-9)

        # 先生成每个点的屏幕坐标
        x0 = sx(u_o)
        y0 = sy(v_o)
        x1 = sx(u_c)
        y1 = sy(v_c)

        # 箭头按深度从远到近绘制（w 越大越近），保证近处的箭头 / 点覆盖在上面。
        order = np.argsort(w_c)
        w_lo = float(w_c.min())
        w_span = max(float(w_c.max()) - w_lo, 1e-9)

        for i in order:
            t = float(min(disp[i] / ref, 1.0))
            color = _lerp_color(t)

            if disp[i] <= move_thr:
                # 点的半径也随透视因子缩放，近处的点更大、远处更小。
                self._draw_point(p, x1[i], y1[i], color, float(f_c[i]))
                continue

            # 3D 位移箭头：从初始坐标 (x0,y0) 直接指向当前三维坐标 (x1,y1)。
            # 起点 / 终点都是经过正交旋转 + 透视除法的 3D 位置，直线方向本就同时
            # 包含 x/y/z 三个分量的贡献（z 由仰角映射为竖直偏移）。不做长度缩放，
            # 因此箭头终点永远正好落在当前点上，不会出现「箭头比当前点短 /
            # 当前点比箭头高」的错位。
            du = x1[i] - x0[i]
            dv = y1[i] - y0[i]
            length = math.hypot(du, dv)
            if length < 1e-6:
                # 位移方向在屏幕上退化（几乎沿视线），用圆点表示当前点
                self._draw_point(p, x1[i], y1[i], color, float(f_c[i]))
                continue

            # 越靠近观察者的箭头画得越粗，强化 3D 纵深感。
            near = (float(w_c[i]) - w_lo) / w_span
            self._draw_arrow(p, x0[i], y0[i], x1[i], y1[i], color, near)
            self._draw_origin_dot(p, x0[i], y0[i], color, float(f_o[i]))
            # 有箭头时也把当前点画出来，颜色与箭头一致
            self._draw_point(p, x1[i], y1[i], color, float(f_c[i]))

        # z 数值标签：在每个当前点旁标出该点的 z 坐标，方便读取真实数值。
        # 点很多时只标出「发生位移」的点，避免标签过密互相遮挡。
        # if self._show_z_labels:
        #     self._draw_z_labels(p, x1, y1, cur, disp, move_thr)

        # 红色半透明箭头：以点云系统原点（包围盒中心，投影到屏幕中心）为起点，
        # 方向 / 长度由 delta_xyz 的平均位移决定（总位移 / 点数），放大显示，
        # 便于观察整体运动方向。半透明红色，渲染尺寸更大更醒目。
        # 若未提供 delta_xyz，则退化为使用点云自身的三维位移 (current - origin)，
        # 保证与点云一致、包含 z 轴并随旋转更新。
        _dd = self._delta_xyz
        if _dd is None and self._origin is not None and self._current is not None:
            try:
                _o = self._origin
                _c = self._current
                _n = min(_o.shape[0], _c.shape[0])
                _dd = _c[:_n] - _o[:_n]
            except Exception:
                _dd = None
        if _dd is not None and _dd.size:
            dd = _dd
            if dd.ndim == 2 and dd.shape[1] == 3 and dd.shape[0] > 0:
                vec = dd.mean(axis=0)
            elif dd.ndim == 1 and dd.size == 3:
                vec = np.asarray(dd, dtype=np.float64).reshape(3)
            else:
                vec = None
            if vec is not None and np.all(np.isfinite(vec)) and float(np.linalg.norm(vec)) > _DELTA_ARROW_VISIBLE_THRESHOLD:
                vec_disp = vec.copy()
                vec_disp[2] *= _Z_SCALE
                arr_disp = vec_disp.reshape(1, 3)
                arr_depth = vec.reshape(1, 3)
                u_tip, v_tip, _, _ = self._project_perspective(
                    arr_disp, arr_depth, self._azimuth, self._elevation, cam_dist
                )
                # "大一点"：放大显示，保证可见
                amp = 100.0
                tip_x = cx + float(u_tip[0]) * scale * amp
                tip_y = cy + float(v_tip[0]) * scale * amp
                dx = tip_x - cx
                dy = tip_y - cy
                length = math.hypot(dx, dy)
                # 最小长度保证可见
                if length < 20.0:
                    if length > 1e-9:
                        tip_x = cx + dx / length * 40.0
                        tip_y = cy + dy / length * 40.0
                    else:
                        tip_x = cx + 40.0
                        tip_y = cy
                    length = math.hypot(tip_x - cx, tip_y - cy)
                # 最大长度避免出框
                max_len = min(rect.width(), rect.height()) * 0.45
                if length > max_len and length > 1e-9:
                    s = max_len / length
                    tip_x = cx + (tip_x - cx) * s
                    tip_y = cy + (tip_y - cy) * s
                # 粗细随合力（位移向量模长）变化：默认与原来的线宽一致，
                # 合力越大越粗，用 sqrt 压缩避免小幅波动就剧烈变化。
                # 最大值在此修改：7.0 是基础粗细，末尾 9.0 是「额外可增粗的上限」，
                # 改大即可让箭头更粗（改小则更收敛）。
                strength = float(np.linalg.norm(vec))
                grow = math.sqrt(max(strength / _DELTA_ARROW_VISIBLE_THRESHOLD, 1.0)) - 1.0
                thickness = 7.0 + min(grow * 1.8, 20.0)
                self._draw_delta_arrow(p, cx, cy, tip_x, tip_y, thickness)

        p.end()

    @staticmethod
    def _draw_point(
        p: QPainter, x: float, y: float, color: QColor, size: float = 1.0
    ) -> None:
        # size 为透视因子（近处 >1、远处 <1），略作限幅避免极端远近时点过大 / 过小。
        size = min(max(size, 0.6), 1.7)
        p.setPen(Qt.PenStyle.NoPen)
        glow = QColor(color)
        glow.setAlpha(46)
        p.setBrush(glow)
        p.drawEllipse(QPointF(x, y), 5.0 * size, 5.0 * size)

        p.setBrush(color)
        p.drawEllipse(QPointF(x, y), 2.2 * size, 2.2 * size)

        core = QColor(255, 255, 255, 180)
        p.setBrush(core)
        p.drawEllipse(QPointF(x, y), 0.8 * size, 0.8 * size)

    def _draw_arrow(
        self,
        p: QPainter,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: QColor,
        near: float,
    ) -> None:
        """从初始坐标 (x0,y0) 指向当前三维坐标 (x1,y1) 的一条直线 3D 位移箭头。

        起点 / 终点均为经过正交投影的 3D 位置，因此直线方向已同时包含 x/y/z
        三个分量的贡献；末端带箭头头部。
        """
        pen = QPen(color, 1.2 + 1.0 * near)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        du = x1 - x0
        dv = y1 - y0
        length = math.hypot(du, dv)
        if length > 1e-6:
            self._draw_arrow_head(p, x1, y1, du, dv, length, color)

    @staticmethod
    def _draw_z_labels(
        p: QPainter,
        xs: np.ndarray,
        ys: np.ndarray,
        cur: np.ndarray,
        disp: np.ndarray,
        move_thr: float,
    ) -> None:
        """在每个当前点旁标出该点的 z 坐标数值。

        ``xs`` / ``ys`` 为当前点的屏幕坐标；``cur`` 为原始 (N,3) 当前点数组，
        取其第 2 列作为真实 z 值。点很多（>120）时只标出发生位移的点，
        避免标签过密互相遮挡。
        """
        n = xs.shape[0]
        label_all = n <= 120

        font = p.font()
        font.setPixelSize(9)
        p.setFont(font)

        p.setPen(QPen(QColor(190, 215, 235, 225)))
        for i in range(n):
            if not label_all and disp[i] <= move_thr:
                continue
            z = float(cur[i, 2])
            p.drawText(QPointF(xs[i] + 5.0, ys[i] - 4.0), f"z={z:.1f}")
        p.setPen(Qt.PenStyle.NoPen)

    @staticmethod
    def _draw_origin_dot(
        p: QPainter, x: float, y: float, color: QColor, size: float = 1.0
    ) -> None:
        size = min(max(size, 0.6), 1.7)
        p.setPen(Qt.PenStyle.NoPen)
        dim = QColor(color)
        dim.setAlpha(150)
        p.setBrush(dim)
        p.drawEllipse(QPointF(x, y), 1.3 * size, 1.3 * size)

    @staticmethod
    def _draw_arrow_head(
        p: QPainter,
        ex: float,
        ey: float,
        du: float,
        dv: float,
        length: float,
        color: QColor,
    ) -> None:
        head = min(max(5.0, length * 0.42), 10.0)
        half = head * 0.5
        ux, uy = du / length, dv / length
        # 垂直方向
        px, py = -uy, ux

        bx = ex - ux * head
        by = ey - uy * head
        poly = QPolygonF(
            [
                QPointF(ex, ey),
                QPointF(bx + px * half, by + py * half),
                QPointF(bx - px * half, by - py * half),
            ]
        )
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawPolygon(poly)

    @staticmethod
    def _draw_delta_arrow(
        p: QPainter,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        thickness: float = 7.0,
    ) -> None:
        """以点云系统原点为起点的红色半透明整体位移箭头。

        ``thickness`` 为箭杆粗细，由调用方按合力大小算好后传入。
        形状为「收腰箭杆 + 实心三角箭头」，带外发光、描边和高光，
        比单纯的粗线段更像一支箭头。
        """
        du = x1 - x0
        dv = y1 - y0
        length = math.hypot(du, dv)
        if length < 1e-6:
            return

        ux, uy = du / length, dv / length
        px, py = -uy, ux  # 单位法向量

        shaft_w = min(max(float(thickness), 4.0), 30.0)
        head_len = min(max(shaft_w * 2.0, 18.0), length * 0.55)
        head_w = shaft_w * 2.6
        shaft_half = shaft_w * 0.5
        head_half = head_w * 0.5
        base_x = x1 - ux * head_len
        base_y = y1 - uy * head_len

        # 1) 柔和外发光：用略粗的圆头线段打底，让箭头更立体
        p.setPen(
            QPen(
                QColor(255, 70, 70, 45),
                shaft_w + 9.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(x0, y0), QPointF(base_x, base_y))

        # 2) 收腰箭杆：尾部略窄、靠近箭头处略宽的梯形，半透明填充 + 深色描边
        tail_narrow = 0.82
        shaft_poly = QPolygonF(
            [
                QPointF(
                    x0 + px * shaft_half * tail_narrow,
                    y0 + py * shaft_half * tail_narrow,
                ),
                QPointF(base_x + px * shaft_half, base_y + py * shaft_half),
                QPointF(base_x - px * shaft_half, base_y - py * shaft_half),
                QPointF(
                    x0 - px * shaft_half * tail_narrow,
                    y0 - py * shaft_half * tail_narrow,
                ),
            ]
        )
        p.setPen(QPen(QColor(165, 20, 20, 170), 1.2))
        p.setBrush(QColor(255, 64, 64, 150))
        p.drawPolygon(shaft_poly)

        # 3) 实心三角箭头：颜色更浓，覆盖在箭杆前端
        head_poly = QPolygonF(
            [
                QPointF(base_x + px * head_half, base_y + py * head_half),
                QPointF(x1, y1),
                QPointF(base_x - px * head_half, base_y - py * head_half),
            ]
        )
        p.setPen(QPen(QColor(150, 18, 18, 200), 1.2))
        p.setBrush(QColor(255, 44, 44, 220))
        p.drawPolygon(head_poly)

    # ------------------------------------------------------------ 交互
    def _can_interact(self) -> bool:
        return self._current is not None and self._current.size > 0

    def mousePressEvent(self, event) -> None:
        if not self._can_interact():
            return
        self._button = event.button()
        self._last_pos = event.position()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.setFocus()

    def mouseMoveEvent(self, event) -> None:
        if self._last_pos is None or not self._can_interact():
            return
        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos
        if self._button == Qt.MouseButton.LeftButton:
            # 水平拖拽：在点云平面内转方位角；竖直拖拽：抬高 / 压低观察角（仰角）。
            self._azimuth = (self._azimuth + dx * 0.008) % (2.0 * math.pi)
            limit = math.radians(88.0)
            self._elevation = float(
                np.clip(self._elevation + dy * 0.008, -limit, limit)
            )
            self.update()
        elif self._button == Qt.MouseButton.RightButton:
            # 右键拖拽：按屏幕像素平移视图（与 depth_3d_view 一致）。
            self._pan_x += dx
            self._pan_y += dy
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._last_pos = None
        self._button = Qt.MouseButton.NoButton
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:
        if not self._can_interact():
            return
        self.reset_view()

    def wheelEvent(self, event) -> None:
        if not self._can_interact():
            return
        delta = event.angleDelta().y()
        # 向上滚（delta>0）放大
        self._zoom = float(np.clip(self._zoom * math.exp(delta * 0.0012), 0.2, 6.0))
        self.update()
