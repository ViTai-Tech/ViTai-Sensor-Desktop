# -*- coding: utf-8 -*-
"""实时深度图 3D 渲染控件（PyQt6 + moderngl，离屏渲染）。

从 ``app/test_depth.py`` 的渲染器改造而来：
- 深度图 -> 高程曲面（亮度/深度 = 高度），逐顶点着色 + 稀疏网格线；
- 用 QOffscreenSurface + QOpenGLContext 建一个共享离屏 GL 上下文，
  渲染到 FBO 后读回成 QPixmap 显示。相比内嵌 QOpenGLWidget，
  这种方式不受 QSS（透明背景 / 圆角 / 堆叠顺序）影响，Windows 上更稳；
- 多个传感器共用一个 GL 上下文，各自持有独立的 VBO/FBO；
- 每帧只更新顶点高度与颜色，几何拓扑按分辨率只建一次；
- 左键旋转、右键平移、双击复位；
- 如果离屏 GL 创建失败，自动退化为 2D 深度色图，保证不会一片空白。

渲染参数与 test_depth.py 保持一致（同样的色带、双面光照），取景按曲面在视口内的
投影包围盒做 contain 适配（宽、高两方向取较紧的一边铺满），与 point_cloud_view 一致。
"""

from __future__ import annotations

import math

import cv2
import moderngl
import numpy as np
from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import (
    QImage,
    QOffscreenSurface,
    QOpenGLContext,
    QPixmap,
    QSurfaceFormat,
)
from PyQt6.QtWidgets import QLabel, QSizePolicy


# --------------------------------------------------------------------- 颜色 / 几何
# 深度色带：深蓝 -> 蓝 -> 青 -> 绿 -> 黄绿 -> 暖黄 -> 近白
_STOPS = np.array(
    [
        [0.07, 0.10, 0.24],
        [0.05, 0.18, 0.42],
        [0.05, 0.44, 0.53],
        [0.18, 0.68, 0.46],
        [0.60, 0.84, 0.29],
        [0.97, 0.90, 0.36],
        [1.00, 1.00, 0.92],
    ],
    dtype=np.float64,
)


# 低值拉伸系数：<1 时小高度会铺满更大范围的色带，低洼处也能看到更丰富的蓝绿层次。
DEPTH_COLOR_GAMMA = 0.4


def depth_colormap(t: np.ndarray) -> np.ndarray:
    """把 0..1 的深度值映射为 RGB 浮点颜色 (N,3)。"""
    seg = len(_STOPS) - 1
    x = (np.clip(t, 0.0, 1.0) ** DEPTH_COLOR_GAMMA) * seg
    lo = np.clip(x.astype(np.int64), 0, seg - 1)
    frac = (x - lo)[..., None]
    rgb = _STOPS[lo] * (1.0 - frac) + _STOPS[lo + 1] * frac
    return rgb.astype("f4")


def build_mesh(depth: np.ndarray, spacing: float = 1.0,
               point_step: int = 4, wire_step: int = 6):
    """把深度数组构建为网格几何数据（同 test_depth.build_mesh）。"""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    x = (xs - (w - 1) * 0.5) * spacing
    # 行0 -> -z（屏幕上方），行末 -> +z（屏幕下方），使顶视与 2D 深度图
    # （行0在上、列0在左）一致；列方向(x)不变，不会水平镜像。
    z = (ys - (h - 1) * 0.5) * spacing
    y = depth
    pos = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype("f4")
    colors = depth_colormap(depth.reshape(-1))

    r = np.arange(h - 1)[:, None]
    c = np.arange(w - 1)[None, :]
    i00 = r * w + c
    i10 = i00 + w
    i11 = i10 + 1
    i01 = i00 + 1
    tris = np.concatenate(
        [
            np.stack([i00, i10, i11], -1).reshape(-1, 3),
            np.stack([i00, i11, i01], -1).reshape(-1, 3),
        ],
        axis=0,
    ).astype("i4")

    edges = []
    for rr in range(0, h, wire_step):
        idx = rr * w + np.arange(w - 1)
        edges.append(np.stack([idx, idx + 1], -1))
    for cc in range(0, w, wire_step):
        idx = np.arange(h - 1) * w + cc
        edges.append(np.stack([idx, idx + w], -1))
    wire = np.concatenate(edges, axis=0).astype("i4")

    if point_step > 1:
        rr = np.arange(h)[::point_step]
        cc = np.arange(w)[::point_step]
        sel = (rr[:, None] * w + cc[None, :]).reshape(-1)
    else:
        sel = np.arange(h * w)

    return pos, colors, tris, wire, pos[sel].copy(), colors[sel].copy()


# SDK 返回的 DEPTH_MAP 实际量程约为 0~1.5（与 2D 深度显示的 vmin/vmax 一致），
# 不是 0~255。按 DEPTH_NORM_MAX 归一化到 0~1 后，再由着色器用
# u_height(=DEPTH_HEIGHT_COEFF) 乘一次得到场景高度。
# DEPTH_HEIGHT_COEFF 越大，起伏越明显；如觉得太平/太陡，直接调这个值即可。
DEPTH_NORM_MAX = 1.5
DEPTH_HEIGHT_COEFF = 100.0


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """把深度图按 ``DEPTH_NORM_MAX`` 线性归一化到 0~1，屏蔽 NaN/Inf。

    深度图的像素值本身就是高度（量程约 0~1.5，非 0~255）：直接按 ``DEPTH_NORM_MAX``
    归一化即可，不做分位数拉伸、不做截断、不做跨帧平滑。这样高度与真实数值
    **严格成正比**（数值大 → 高度大），既不会出现“顶部平面”，也不会产生迟滞或
    反直觉的缩放。最终场景高度由着色器的 ``u_height``（= DEPTH_HEIGHT_COEFF）再乘一次。
    """
    d = np.asarray(depth, dtype=np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(d, 0.0, DEPTH_NORM_MAX) / DEPTH_NORM_MAX
    return out.astype("f4")


def _downsample(depth: np.ndarray, max_side: int) -> np.ndarray:
    """按步长降采样，控制顶点数（多传感器同时显示时避免 CPU/GPU 压力过大）。"""
    h, w = depth.shape
    step = max(1, int(math.ceil(max(h, w) / float(max_side))))
    if step == 1:
        return depth
    return depth[::step, ::step]


def _default_view_basis():
    """默认朝向下的相机基向量，返回 ``(eye_dir, up, right)``。

    与 ``OrbitCamera`` 的构造保持一致（``Rz(tilt_z) @ Rx(tilt_x)`` 作用于
    ``_EYE_DIR0`` / ``_UP_0``），供取景时按默认朝向计算投影包围盒。
    """
    ax, az = _DEFAULT_TILT_X, _DEFAULT_TILT_Z
    cx, sx = math.cos(ax), math.sin(ax)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cx, -sx],
                   [0.0, sx, cx]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0],
                   [sz, cz, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    rot = rz @ rx
    eye_dir = rot @ np.array([0.0, 1.0, 0.0])
    up = rot @ np.array([0.0, 0.0, -1.0])
    forward = -eye_dir
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, forward)
    return eye_dir, up, right


# --------------------------------------------------------------------- 相机 / 矩阵
def _mat_bytes(m: np.ndarray) -> bytes:
    """把行主序的数学矩阵转成 moderngl 需要的列主序字节。"""
    return np.asarray(m, dtype="f4").T.tobytes()


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = _normalize(target - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


class OrbitCamera:
    """以 target 为中心的 3D 倾斜相机。

    用两个互相垂直的倾斜角描述相机的观察方向，二者都围绕 target（3D 图中心）
    旋转，因此上下、左右拖拽都会像“绕中心翘起曲面”一样观察 3D 起伏，而不是在
    平面内打转：

    - ``tilt_x``：绕世界 X 轴倾斜（上下拖拽），相机移到 ±Z 一侧；
    - ``tilt_z``：绕世界 Z 轴倾斜（左右拖拽），相机移到 ±X 一侧。

    与原来的 theta/phi 轨道相机不同，这里不再有“正上方时左右自转退化为平面内
    旋转”的问题：默认就是正俯视，左右拖拽即绕 Z 轴把相机抬向一侧。
    """

    # 倾斜上限略小于 90°，保证相机始终位于曲面上方、不会翻转到背面。
    TILT_LIMIT = math.pi * 0.5 - 0.02

    # 基准朝向：相机在正上方俯视；屏幕上 = -Z，屏幕右 = +X（与 2D 深度图一致）。
    _EYE_DIR0 = np.array([0.0, 1.0, 0.0])
    _UP_0 = np.array([0.0, 0.0, -1.0])

    def __init__(self, target=(0.0, 0.0, 0.0), distance=10.0,
                 tilt_x=0.0, tilt_z=0.0, fov=40.0):
        self.target = np.array(target, dtype=np.float64)
        self.distance = float(distance)
        self.tilt_x = float(tilt_x)
        self.tilt_z = float(tilt_z)
        self.fov = float(fov)
        self.near = 0.1
        self.far = 5000.0

    def _rotation(self) -> np.ndarray:
        """绕世界 X（tilt_x）与 Z（tilt_z）轴的复合旋转矩阵。"""
        ax, az = self.tilt_x, self.tilt_z
        cx, sx = math.cos(ax), math.sin(ax)
        cz, sz = math.cos(az), math.sin(az)
        rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, cx, -sx],
                       [0.0, sx, cx]], dtype=np.float64)
        rz = np.array([[cz, -sz, 0.0],
                       [sz, cz, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
        return rz @ rx

    def eye(self) -> np.ndarray:
        return self.target + self.distance * (self._rotation() @ self._EYE_DIR0)

    def up(self) -> np.ndarray:
        return self._rotation() @ self._UP_0

    def view_projection(self, aspect: float) -> np.ndarray:
        # up 由同一个旋转矩阵得到，始终与视线正交，因此任何倾斜角下都不会退化。
        view = _look_at(self.eye(), self.target, self.up())
        proj = _perspective(self.fov, aspect, self.near, self.far)
        return proj @ view

    def tilt(self, ddx: float, ddy: float) -> None:
        """左右/上下拖拽 -> 绕 Z / X 轴倾斜。

        方向与鼠标拖拽保持一致：向下拖拽（ddy>0）画面向下、向左拖拽（ddx<0）
        画面向左，与直觉一致。
        """
        limit = self.TILT_LIMIT
        self.tilt_x = float(np.clip(self.tilt_x - ddy, -limit, limit))
        self.tilt_z = float(np.clip(self.tilt_z + ddx, -limit, limit))

    def pan(self, dx: float, dy: float, viewport_h: float) -> None:
        """屏幕像素位移 -> 相机平面平移。"""
        eye = self.eye()
        up = self.up()
        forward = _normalize(self.target - eye)
        right = _normalize(np.cross(forward, up))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        up = np.cross(right, forward)
        scale = self.distance / max(viewport_h, 1.0)
        self.target += (-dx * right + dy * up) * scale


# --------------------------------------------------------------------- GLSL
_SURFACE_VS = """
#version 330
uniform mat4 u_mvp;
uniform float u_height;
in vec3 in_pos;
in vec3 in_color;
out vec3 v_world;
out vec3 v_color;
void main() {
    vec3 p = in_pos;
    p.y *= u_height;
    v_world = p;
    v_color = in_color;
    gl_Position = u_mvp * vec4(p, 1.0);
}
"""

_SURFACE_FS = """
#version 330
in vec3 v_world;
in vec3 v_color;
out vec4 f_color;
uniform vec3 u_eye;
uniform vec3 u_light;
void main() {
    vec3 N = normalize(cross(dFdx(v_world), dFdy(v_world)));
    vec3 L = normalize(u_light);
    vec3 V = normalize(u_eye - v_world);
    float ndl = dot(N, L);
    if (ndl < 0.0) { N = -N; ndl = -ndl; }
    float diff = max(ndl, 0.0);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 42.0);
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.5);
    // 俯视时法线接近灯光方向，漫反射接近满值，避免整片被冲亮：
    // 降低漫反射权重与高光强度，保留底色并增加明暗层次。
    vec3 col = v_color * (0.22 + 0.62 * diff);
    col += vec3(1.0) * spec * 0.22;
    col += v_color * rim * 0.30;
    f_color = vec4(col, 1.0);
}
"""

_WIRE_VS = """
#version 330
uniform mat4 u_mvp;
uniform float u_height;
in vec3 in_pos;
void main() {
    vec3 p = in_pos;
    p.y *= u_height;
    vec4 clip = u_mvp * vec4(p, 1.0);
    clip.z -= 0.0015 * clip.w;
    gl_Position = clip;
}
"""

_WIRE_FS = """
#version 330
out vec4 f_color;
uniform vec4 u_color;
void main() { f_color = u_color; }
"""

_POINT_VS = """
#version 330
uniform mat4 u_mvp;
uniform float u_height;
uniform float u_size;
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
void main() {
    vec3 p = in_pos;
    p.y *= u_height;
    gl_Position = u_mvp * vec4(p, 1.0);
    gl_PointSize = u_size;
    v_color = in_color;
}
"""

_POINT_FS = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r = length(d);
    if (r > 1.0) discard;
    float a = 1.0 - smoothstep(0.45, 1.0, r);
    f_color = vec4(v_color * a, a);
}
"""

_LIGHT_DIR = (0.35, 0.9, 0.35)

# 默认视角：与 point_cloud_view 对齐，相机沿 X 轴抬起 25°（稍带俯视的立体感，
# 而不是正俯视整张深度面），左右倾斜角仍为 0。
_DEFAULT_TILT_X = math.radians(25.0)
_DEFAULT_TILT_Z = 0.0

# 相机取景留白系数：1.0 时曲面刚好铺满外框（宽、高两方向中较紧的一边），不留黑边。
_FIT_MARGIN = 1.0

class DepthScene:
    """把深度网格数据上传到 GPU 并负责绘制（与 Qt 无关）。"""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.prog_surface = ctx.program(vertex_shader=_SURFACE_VS, fragment_shader=_SURFACE_FS)
        self.prog_wire = ctx.program(vertex_shader=_WIRE_VS, fragment_shader=_WIRE_FS)
        self.prog_point = ctx.program(vertex_shader=_POINT_VS, fragment_shader=_POINT_FS)

        self.prog_surface["u_light"].value = _LIGHT_DIR
        self.prog_wire["u_color"].value = (0.68, 0.92, 1.0, 0.28)

        self._shape = None
        self.vbo_pos = self.vbo_col = self.ibo_tri = self.ibo_wire = None
        self.vbo_ppt = self.vbo_pcol = None
        self.vao_surface = self.vao_wire = self.vao_points = None
        self.n_tri_idx = self.n_wire_idx = self.n_pts = 0

    def _release_geometry(self) -> None:
        for obj in (self.vao_surface, self.vao_wire, self.vao_points,
                    self.vbo_pos, self.vbo_col, self.ibo_tri, self.ibo_wire,
                    self.vbo_ppt, self.vbo_pcol):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self.vao_surface = self.vao_wire = self.vao_points = None
        self.vbo_pos = self.vbo_col = self.ibo_tri = self.ibo_wire = None
        self.vbo_ppt = self.vbo_pcol = None

    def set_depth(self, depth: np.ndarray) -> None:
        """上传或更新深度（0..1）。拓扑不变时只重写顶点数据。"""
        if depth.ndim != 2 or depth.size == 0:
            return
        pos, colors, tris, wire, ppts, pcols = build_mesh(depth)

        if self._shape != depth.shape:
            self._release_geometry()
            self._shape = depth.shape
            ctx = self.ctx
            self.vbo_pos = ctx.buffer(pos.tobytes())
            self.vbo_col = ctx.buffer(colors.tobytes())
            self.ibo_tri = ctx.buffer(tris.ravel().tobytes())
            self.vao_surface = ctx.vertex_array(
                self.prog_surface,
                [(self.vbo_pos, "3f", "in_pos"), (self.vbo_col, "3f", "in_color")],
                index_buffer=self.ibo_tri,
            )
            self.ibo_wire = ctx.buffer(wire.ravel().tobytes())
            self.vao_wire = ctx.vertex_array(
                self.prog_wire,
                [(self.vbo_pos, "3f", "in_pos")],
                index_buffer=self.ibo_wire,
            )
            self.vbo_ppt = ctx.buffer(ppts.tobytes())
            self.vbo_pcol = ctx.buffer(pcols.tobytes())
            self.vao_points = ctx.vertex_array(
                self.prog_point,
                [(self.vbo_ppt, "3f", "in_pos"), (self.vbo_pcol, "3f", "in_color")],
            )
        else:
            self.vbo_pos.write(pos.tobytes())
            self.vbo_col.write(colors.tobytes())
            self.vbo_ppt.write(ppts.tobytes())
            self.vbo_pcol.write(pcols.tobytes())

        self.n_tri_idx = tris.size
        self.n_wire_idx = wire.size
        self.n_pts = ppts.shape[0]

    def render(self, mvp, eye, height: float, *,
               show_surface=True, show_wire=True, show_points=False,
               size_scale: float = 1.0) -> None:
        if self.vao_surface is None:
            return
        ctx = self.ctx
        mvp_b = _mat_bytes(mvp)

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_mask = True

        if show_surface:
            self.prog_surface["u_mvp"].write(mvp_b)
            self.prog_surface["u_height"].value = float(height)
            self.prog_surface["u_eye"].value = tuple(float(v) for v in eye)
            self.vao_surface.render(moderngl.TRIANGLES, vertices=self.n_tri_idx)

        if show_wire:
            self.prog_wire["u_mvp"].write(mvp_b)
            self.prog_wire["u_height"].value = float(height)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            ctx.depth_mask = False
            self.vao_wire.render(moderngl.LINES, vertices=self.n_wire_idx)
            ctx.depth_mask = True
            ctx.disable(moderngl.BLEND)

        if show_points:
            self.prog_point["u_mvp"].write(mvp_b)
            self.prog_point["u_height"].value = float(height)
            self.prog_point["u_size"].value = 3.2 * size_scale
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            ctx.depth_mask = False
            self.vao_points.render(moderngl.POINTS, vertices=self.n_pts)
            ctx.depth_mask = True
            ctx.disable(moderngl.BLEND)

    def release(self) -> None:
        self._release_geometry()
        for obj in (self.prog_surface, self.prog_wire, self.prog_point):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass


# --------------------------------------------------------------------- 离屏 GL 上下文
class _OffscreenGL:
    """进程内共享的离屏 OpenGL 上下文（在 GUI 线程按需初始化）。"""

    def __init__(self) -> None:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        self.surface = QOffscreenSurface()
        self.surface.setFormat(fmt)
        self.surface.create()
        if not self.surface.isValid():
            raise RuntimeError("无法创建离屏 OpenGL surface")

        self.context = QOpenGLContext()
        self.context.setFormat(fmt)
        if not self.context.create():
            raise RuntimeError("无法创建 OpenGL 上下文")
        if not self.context.makeCurrent(self.surface):
            raise RuntimeError("无法激活 OpenGL 上下文")

        self.ctx = moderngl.create_context()
        # 不留占用：需要时各控件自己 makeCurrent
        self.context.doneCurrent()

    def make_current(self) -> None:
        if not self.context.makeCurrent(self.surface):
            raise RuntimeError("无法激活 OpenGL 上下文")

    def done_current(self) -> None:
        self.context.doneCurrent()


_OFFSCREEN_GL: _OffscreenGL | None = None
_OFFSCREEN_GL_FAILED = False


def _get_offscreen_gl() -> _OffscreenGL:
    global _OFFSCREEN_GL, _OFFSCREEN_GL_FAILED
    if _OFFSCREEN_GL is not None:
        return _OFFSCREEN_GL
    if _OFFSCREEN_GL_FAILED:
        raise RuntimeError("离屏 OpenGL 不可用")
    try:
        _OFFSCREEN_GL = _OffscreenGL()
    except Exception as exc:  # 没有可用 GL 时退化到 2D
        _OFFSCREEN_GL_FAILED = True
        print(f"[depth3d] 离屏 OpenGL 初始化失败，已退化为 2D 显示: {exc}")
        raise
    return _OFFSCREEN_GL


# 外框与加载态样式统一对齐 point_cloud_view：同一套外框（深底 + 蓝边 + 圆角），
# 正常态文字为灰色，加载态文字为青色，外框保持不变。
_DEPTH_IDLE_STYLE = (
    "background-color: #070b14;"
    "border: 1px solid #1a3a5c;"
    "border-radius: 8px;"
    "color: #475569;"
)
_DEPTH_LOADING_STYLE = (
    "background-color: #070b14;"
    "border: 1px solid #1a3a5c;"
    "border-radius: 8px;"
    "color: #67e8f9;"
)


# --------------------------------------------------------------------- Qt 控件
class Depth3DWidget(QLabel):
    """深度曲面控件：离屏渲染成图片显示，左键旋转、右键平移、双击复位。

    对外接口与 ImageLabel 对齐（set_depth / set_loading / clear），
    可直接替换原来的 2D 深度图控件。
    """

    MAX_SIDE = 128           # 参与网格构建的最大边长（降采样后）
    MAX_RENDER = 900         # 离屏渲染最大边长（像素）
    RENDER_INTERVAL_MS = 90  # 刷新上限（深度本身约 8Hz，没必要更高）

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(100, 70)
        # 与 point_cloud_view 对齐：占满父布局可用空间，全屏 / 拉宽外层容器时
        # 控件自身才会被撑大，绘制区才能跟着变宽。
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setStyleSheet(_DEPTH_IDLE_STYLE)
        self.setText("无信号")
        self._loading_style = False

        self._gl: _OffscreenGL | None = None
        self._scene: DepthScene | None = None
        self._fbo = None
        self._fbo_color = None
        self._fbo_depth = None
        self._fbo_size = None
        self._gl_failed = False

        self._depth: np.ndarray | None = None
        self._signature = None
        self._version = 0
        self._scene_version = -1
        self._dirty = False

        self._depth_size = (1, 1)
        self.height_scale = 70.0
        # 旋转中心 = 当前深度曲面的包围盒中心；用户手动平移后不再自动跟随数据。
        self._center = np.array([0.0, 0.0, 0.0])
        self._panned = False
        # 取景用：默认朝向下每个顶点的投影分量（沿 right / up / 视线），
        # 深度更新时算一次；每帧只需按当前宽高比反解相机距离。
        self._fit_xv = None
        self._fit_yv = None
        self._fit_along = None
        self.camera = OrbitCamera(
            distance=10.0,
            tilt_x=_DEFAULT_TILT_X, tilt_z=_DEFAULT_TILT_Z, fov=40.0,
        )
        self._zoom = 1.0           # 滚轮缩放系数（1.0 = 自适应满屏）

        self._button = Qt.MouseButton.NoButton
        self._last_pos = None

        self._timer = QTimer(self)
        self._timer.setInterval(self.RENDER_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    # ----------------------------------------------------------- 尺寸策略
    def sizeHint(self):
        return QSize(160, 120)

    def minimumSizeHint(self):
        return QSize(100, 70)

    # ----------------------------------------------------------- public API
    def set_depth(self, depth) -> None:
        """外部每收到一帧深度图调用一次。"""
        if depth is None:
            return
        try:
            raw = np.asarray(depth)
            if raw.ndim != 2 or raw.size == 0:
                return
            # 轻量帧里 merged 会重复携带同一帧深度对象；用采样指纹去重，
            # 同时能识别 SDK 复用同一块缓冲区并就地改写的情况。
            sample = raw[::16, ::16]
            sig = (raw.shape, str(raw.dtype), sample.tobytes())
            if sig == self._signature:
                return
            self._signature = sig
            arr = raw.astype(np.float32, copy=False)
            arr = _downsample(arr, self.MAX_SIDE)
            arr = normalize_depth(arr)
        except Exception as exc:
            print(f"[depth3d] 深度数据预处理失败: {exc}")
            return

        self._depth = arr
        resized = self._prepare_geometry()
        self._recenter_target(force=resized)
        self._version += 1
        self._dirty = True

    def clear(self) -> None:
        self._depth = None
        self._signature = None
        self._dirty = False
        self.clear_display()

    def set_loading(self, loading: bool) -> None:
        if loading:
            self.clear()
            self.setStyleSheet(_DEPTH_LOADING_STYLE)
            self._loading_style = True
            self.setText("加载中...")
        elif self._depth is None:
            self.clear_display()

    def clear_display(self) -> None:
        self.setPixmap(QPixmap())
        self.setStyleSheet(_DEPTH_IDLE_STYLE)
        self._loading_style = False
        self.setText("无信号")

    def _apply_idle_style(self) -> None:
        """数据重新显示后，把加载态的青色文字样式恢复为常规外框。"""
        if self._loading_style:
            self.setStyleSheet(_DEPTH_IDLE_STYLE)
            self._loading_style = False

    def release_resources(self) -> None:
        """行被销毁前调用，释放该控件独占的 GL 资源。"""
        if self._gl is not None:
            try:
                self._gl.make_current()
                self._release_fbo()
                if self._scene is not None:
                    self._scene.release()
                    self._scene = None
                self._gl.done_current()
            except Exception:
                pass
        self._scene = None
        self._fbo = None
        self._fbo_size = None

    def _release_fbo(self) -> None:
        for obj in (self._fbo, self._fbo_color, self._fbo_depth):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self._fbo = None
        self._fbo_color = None
        self._fbo_depth = None
        self._fbo_size = None

    def _make_fbo(self, rw: int, rh: int):
        """创建带深度附件的离屏 FBO（RGBA，背景透明，无渐变背景图）。"""
        ctx = self._gl.ctx
        color = ctx.texture((rw, rh), components=4)
        depth = ctx.depth_renderbuffer((rw, rh))
        self._fbo = ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
        self._fbo_color = color
        self._fbo_depth = depth

    # ----------------------------------------------------------- 事件
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._dirty = True

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._dirty = True

    def _on_tick(self) -> None:
        if self._dirty and self._depth is not None and self.isVisible():
            self._render_now()

    # ----------------------------------------------------------- 渲染
    def _ensure_scene(self) -> bool:
        if self._scene is not None:
            return True
        if self._gl_failed:
            return False
        try:
            self._gl = _get_offscreen_gl()
            self._gl.make_current()
            self._scene = DepthScene(self._gl.ctx)
            self._scene_version = -1
            return True
        except Exception as exc:
            self._gl_failed = True
            self._gl = None
            self._scene = None
            print(f"[depth3d] 3D 渲染初始化失败，已退化为 2D: {exc}")
            return False

    def _fit_distance(self, aspect: float) -> float:
        if self._fit_xv is None:
            return 10.0
        vfov = math.radians(self.camera.fov)
        tan_v = max(math.tan(vfov * 0.5), 1e-6)
        tan_h = max(tan_v * max(aspect, 1e-6), 1e-6)
        # contain 适配：分别求“铺满宽度所需距离”和“铺满高度所需距离”，取较大者
        # （等价于用两者中较小的适配比例），保证整个曲面（含高程起伏）都落在框内。
        d_w = float(np.max(np.abs(self._fit_xv) / tan_h + self._fit_along))
        d_h = float(np.max(np.abs(self._fit_yv) / tan_v + self._fit_along))
        return max(d_w, d_h, 1e-6) * _FIT_MARGIN

    def _render_now(self) -> None:
        if self._depth is None:
            return
        w = self.width()
        h = self.height()
        if w < 8 or h < 8:
            return

        if not self._ensure_scene():
            self._draw_2d()
            return

        dpr = self.devicePixelRatioF()
        # 目标物理像素尺寸（显示用）：始终等于控件实际尺寸 × dpr，保证铺满外框。
        # 控件随外层容器拉伸时 tw/th 随之变大，取景再按当前宽高比做 contain 适配。
        tw = max(8, int(w * dpr))
        th = max(8, int(h * dpr))
        # 实际 FBO 渲染分辨率受 MAX_RENDER 限制（多传感器时控制 GPU 负载），
        # 但保持与控件相同的宽高比；读回后再放大到目标尺寸，做到「渲染省、显示满」。
        rw, rh = tw, th
        if max(rw, rh) > self.MAX_RENDER:
            scale = self.MAX_RENDER / float(max(rw, rh))
            rw = max(8, int(rw * scale))
            rh = max(8, int(rh * scale))

        gl = self._gl
        try:
            gl.make_current()

            if self._fbo is None or self._fbo_size != (rw, rh):
                self._release_fbo()
                self._make_fbo(rw, rh)
                self._fbo_size = (rw, rh)

            if self._scene_version != self._version:
                self._scene.set_depth(self._depth)
                self._scene_version = self._version

            self._fbo.use()
            self._fbo.clear(0.0, 0.0, 0.0, 0.0)  # 透明背景，不绘制背景图

            aspect = rw / float(rh)
            self.camera.distance = self._fit_distance(aspect) * self._zoom
            mvp = self.camera.view_projection(aspect)
            self._scene.render(
                mvp, self.camera.eye(), self.height_scale,
                show_surface=True, show_wire=True, show_points=False,
                size_scale=dpr,
            )

            data = self._fbo.read(components=4)
            gl.done_current()
        except Exception as exc:
            try:
                gl.done_current()
            except Exception:
                pass
            self._gl_failed = True
            print(f"[depth3d] 3D 渲染失败，已退化为 2D: {exc}")
            self._draw_2d()
            return

        img = QImage(data, rw, rh, rw * 4, QImage.Format.Format_RGBA8888).copy()
        img = img.mirrored(False, True)  # OpenGL 原点在左下
        # FBO 分辨率可能受 MAX_RENDER 限制而小于目标尺寸，放大到控件物理尺寸，
        # 保证绘制区铺满外框（宽高比一致，不会变形）。
        if (img.width(), img.height()) != (tw, th):
            img = img.scaled(
                tw, th,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        pix = QPixmap.fromImage(img)
        if dpr > 0:
            # 逻辑尺寸 = 物理尺寸 / dpr = 控件尺寸，正好铺满、且高 DPI 下清晰
            pix.setDevicePixelRatio(dpr)
        self._apply_idle_style()
        self.setPixmap(pix)
        self._dirty = False

    def _draw_2d(self) -> None:
        """无 GL 时的兜底：显示 2D 深度色图，至少不会一片空白。"""
        if self._depth is None:
            self.clear_display()
            return
        arr8 = (np.clip(self._depth, 0.0, 1.0) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(arr8, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(img).scaled(
            self.size(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._apply_idle_style()
        self.setPixmap(pix)
        self._dirty = False

    def _prepare_geometry(self) -> bool:
        w, h = self._depth.shape[1], self._depth.shape[0]
        resized = (w, h) != self._depth_size
        if resized:
            self._depth_size = (w, h)
            # 高度 = 归一化深度(0~1) * 系数；用固定系数做场景高度，
            # 相机按整个曲面包围盒（含高度）做 contain 适配。
            self.height_scale = DEPTH_HEIGHT_COEFF
            self._panned = False
            # 旋转 / 取景中心取曲面包围盒中心：X/Z 居中于原点，竖直方向取高度范围
            # 中点，使 contain 取景后曲面在框内上下居中。只在分辨率变化时计算一次，
            # 不逐帧跟随实时 min/max，避免按压导致曲面起伏时整幅图上下抖动。
            lo = float(np.min(self._depth))
            hi = float(np.max(self._depth))
            self._center = np.array(
                [0.0, 0.5 * (lo + hi) * self.height_scale, 0.0], dtype=np.float64
            )
            # 取景包围盒同样只在分辨率变化时算一次：这样相机距离只随外框宽高比
            # 变化（跟框自适应），而曲面局部起伏（按压）不会让整幅图忽大忽小。
            self._prepare_fit_projection()
        return resized

    def _recenter_target(self, force: bool = False) -> None:
        """把旋转 / 取景中心跟随到曲面包围盒中心（在分辨率变化时更新，见
        ``_prepare_geometry``）。

        竖直方向取高度范围中点，使 contain 取景后曲面在框内上下居中；该中心只在
        分辨率变化时计算一次，不逐帧跟随实时 min/max，避免按压起伏时整幅图抖动。

        非强制更新时，仅在用户没有手动平移、也没有按住鼠标时复位中心；
        一旦用户平移过（``_panned``），就不再自动复位，避免覆盖用户视角。
        """
        if self._depth is None:
            return
        if force or (not self._panned
                     and self._button == Qt.MouseButton.NoButton):
            self.camera.target = self._center.copy()

    def _prepare_fit_projection(self) -> None:
        """预计算默认朝向下每个顶点沿 ``right`` / ``up`` / 视线的分量。

        取景时（``_fit_distance``）只需把这些分量按当前宽高比换算成所需相机
        距离，避免每帧重新投影全部顶点。顶点坐标与 ``build_mesh`` 一致：
        X/Z 居中于原点，Y = 归一化深度 × 高度系数；再相对旋转中心平移。
        """
        if self._depth is None:
            self._fit_xv = self._fit_yv = self._fit_along = None
            return
        h, w = self._depth.shape
        xs = np.arange(w, dtype=np.float64) - (w - 1) * 0.5
        zs = np.arange(h, dtype=np.float64) - (h - 1) * 0.5
        gx, gz = np.meshgrid(xs, zs)
        gy = self._depth.astype(np.float64) * self.height_scale
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        pts = pts - self._center
        eye_dir, up, right = _default_view_basis()
        self._fit_xv = pts @ right
        self._fit_yv = pts @ up
        self._fit_along = pts @ eye_dir

    # ----------------------------------------------------------- 交互
    def _cam_ready(self) -> bool:
        return self._depth is not None

    def mousePressEvent(self, event) -> None:
        if not self._cam_ready():
            return
        self._button = event.button()
        self._last_pos = event.position()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.setFocus()

    def mouseMoveEvent(self, event) -> None:
        if self._last_pos is None or not self._cam_ready():
            return
        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos
        if self._button == Qt.MouseButton.LeftButton:
            self.camera.tilt(dx * 0.008, dy * 0.008)
        elif self._button == Qt.MouseButton.RightButton:
            self.camera.pan(dx, dy, float(self.height()))
            self._panned = True
        self._dirty = True
        self._render_now()

    def mouseReleaseEvent(self, event) -> None:
        self._last_pos = None
        self._button = Qt.MouseButton.NoButton
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:
        if not self._cam_ready():
            return
        self.camera.tilt_x = _DEFAULT_TILT_X
        self.camera.tilt_z = _DEFAULT_TILT_Z
        self._panned = False
        self.camera.target = self._center.copy()
        self._zoom = 1.0
        self._dirty = True
        self._render_now()

    def wheelEvent(self, event) -> None:
        if not self._cam_ready():
            return
        delta = event.angleDelta().y()
        # 向上滚（delta>0）放大：距离缩短
        factor = math.exp(-delta * 0.0012)
        self._zoom = float(np.clip(self._zoom * factor, 0.2, 6.0))
        self._dirty = True
        self._render_now()
