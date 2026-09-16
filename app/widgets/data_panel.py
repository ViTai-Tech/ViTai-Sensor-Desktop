import sys

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import numpy as np
from collections import deque
from pyvitaisdk import VTSDataType


_MONO_FONT = "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"


class ForcePlots(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._buffer = deque(maxlen=300)  # 可见帧：30Hz 下约 10 秒波形历史

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        pg.setConfigOptions(antialias=True)

        self.fx_plot = self._new_plot()
        self.fx_curve = self.fx_plot.plot(pen=pg.mkPen("#e74c3c", width=2))
        self.fx_axis = self.fx_plot.getPlotItem().getAxis("left")
        self.fx_refs = self._new_ref_lines(self.fx_plot, "#f5b7b1")
        self.fx_value = self._new_value_label("#e74c3c")
        self.fy_plot = self._new_plot()
        self.fy_curve = self.fy_plot.plot(pen=pg.mkPen("#2ecc71", width=2))
        self.fy_axis = self.fy_plot.getPlotItem().getAxis("left")
        self.fy_refs = self._new_ref_lines(self.fy_plot, "#a9e6c3")
        self.fy_value = self._new_value_label("#2ecc71")
        self.fz_plot = self._new_plot()
        self.fz_curve = self.fz_plot.plot(pen=pg.mkPen("#3498db", width=2))
        self.fz_axis = self.fz_plot.getPlotItem().getAxis("left")
        self.fz_refs = self._new_ref_lines(self.fz_plot, "#a9d3f0")
        self.fz_value = self._new_value_label("#3498db")

        layout.addWidget(self._labeled_row("Fx", self.fx_plot, self.fx_value), 1)
        layout.addWidget(self._labeled_row("Fy", self.fy_plot, self.fy_value), 1)
        layout.addWidget(self._labeled_row("Fz", self.fz_plot, self.fz_value), 1)

    def _new_plot(self):
        p = pg.PlotWidget()
        p.setBackground("#ffffff")
        p.showGrid(x=False, y=False)   # 关闭默认网格，参考线改由 _new_ref_lines 显式绘制
        pi = p.getPlotItem()
        pi.hideButtons()
        pi.showAxis("left")
        pi.hideAxis("bottom")
        pi.setMouseEnabled(x=False, y=False)

        # 纵坐标（y 轴）：小字号、紧凑宽度；轴线与刻度文字灰色，刻度短线浅灰
        y_axis = pi.getAxis("left")
        y_axis.setStyle(tickFont=QFont(_MONO_FONT, 7))
        y_axis.setWidth(38)
        y_axis.setPen(pg.mkPen("#999999"))        # 轴线灰色
        y_axis.setTextPen(pg.mkPen("#999999"))    # 刻度文字灰色
        y_axis.setTickPen(pg.mkPen("#cccccc"))    # 刻度短线浅灰（非参考线）
        return p

    def _new_ref_lines(self, plot, light_color):
        """三条数值参考线（-峰值 / 0 / +峰值），用各轴颜色的浅色版本，位置随峰值更新。"""
        refs = []
        for _ in range(3):
            ln = pg.InfiniteLine(angle=0, pen=pg.mkPen(light_color, width=1))
            plot.addItem(ln)
            refs.append(ln)
        return refs

    def _new_value_label(self, color):
        lab = QLabel("--")
        lab.setFixedWidth(52)
        lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lab.setFont(QFont(_MONO_FONT, 8, QFont.Weight.Bold))
        lab.setStyleSheet(f"color: {color};")
        return lab

    def _labeled_row(self, name, plot, value_label):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(2)
        lab = QLabel(name)
        lab.setFixedWidth(18)
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lab.setStyleSheet("font-weight: bold; font-size: 10px; color: #666;")
        l.addWidget(lab)
        l.addWidget(plot, 1)
        l.addWidget(value_label)
        return w

    def update_data(self, force6d):
        if force6d is None:
            return
        self._buffer.append(force6d)
        if len(self._buffer) == 0:
            return

        x = list(range(len(self._buffer)))
        arr = np.array(self._buffer)
        self.fx_curve.setData(x, arr[:, 0])
        self.fy_curve.setData(x, arr[:, 1])
        self.fz_curve.setData(x, arr[:, 2])

        # 右侧固定数值
        self.fx_value.setText(f"{arr[-1, 0]:.2f}")
        self.fy_value.setText(f"{arr[-1, 1]:.2f}")
        self.fz_value.setText(f"{arr[-1, 2]:.2f}")

        # 纵坐标：峰值维持到出框——以可见帧内最大绝对值为上下界，0 居中不变
        abs_arr = np.abs(arr)
        self._apply_y_scale(self.fx_plot, self.fx_axis, self.fx_refs, float(abs_arr[:, 0].max()))
        self._apply_y_scale(self.fy_plot, self.fy_axis, self.fy_refs, float(abs_arr[:, 1].max()))
        self._apply_y_scale(self.fz_plot, self.fz_axis, self.fz_refs, float(abs_arr[:, 2].max()))

    def _apply_y_scale(self, plot, axis, refs, v):
        v = max(v, 0.01)   # 防止零范围
        margin = v * 0.2   # 上下留白，避免刻度标签贴边被裁
        plot.setYRange(-v - margin, v + margin, padding=0)
        ticks = [
            (-v, self._fmt(-v)),
            (0.0, "0"),
            (v, self._fmt(v)),
        ]
        axis.setTicks([ticks, []])
        # 数值参考线对齐到当前刻度值
        refs[0].setPos(-v)
        refs[1].setPos(0.0)
        refs[2].setPos(v)

    def _fmt(self, v):
        av = abs(v)
        if av >= 100:
            return f"{v:.0f}"
        if av >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"

    def reset(self):
        self._buffer.clear()
        x = []
        y = []
        self.fx_curve.setData(x, y)
        self.fy_curve.setData(x, y)
        self.fz_curve.setData(x, y)
        for lab in (self.fx_value, self.fy_value, self.fz_value):
            lab.setText("--")
        for axis in (self.fx_axis, self.fy_axis, self.fz_axis):
            axis.setTicks([[], []])
        for refs in (self.fx_refs, self.fy_refs, self.fz_refs):
            for ln in refs:
                ln.setPos(0.0)


class DataPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addWidget(QLabel("力", font=QFont("", 10, QFont.Weight.Bold)))
        self.force_plots = ForcePlots()
        layout.addWidget(self.force_plots)

        slip_group = QGroupBox("滑动检测")
        sg_layout = QHBoxLayout(slip_group)
        self.slip_indicator = QLabel("●")
        self.slip_indicator.setFont(QFont("", 18))
        self.slip_indicator.setStyleSheet("color: #2ecc71")
        self.slip_label = QLabel("--")
        self.slip_label.setFont(QFont("", 14, QFont.Weight.Bold))
        sg_layout.addWidget(self.slip_indicator)
        sg_layout.addWidget(self.slip_label)
        sg_layout.addStretch()
        layout.addWidget(slip_group)

        layout.addStretch()

    def update_data(self, data: dict):
        if VTSDataType.FORCE6D_VECTOR in data:
            f = data[VTSDataType.FORCE6D_VECTOR]
            if f is not None and len(f) >= 6:
                self.force_plots.update_data(f)

        if VTSDataType.SLIP_STATE in data:
            state = data[VTSDataType.SLIP_STATE]
            if state is not None:
                state_name = str(state.name)
                self.slip_label.setText(state_name)
                if "SLIP" in state_name.upper():
                    self.slip_indicator.setStyleSheet("color: #e74c3c")
                    self.slip_label.setStyleSheet("color: #e74c3c")
                elif "UNSTABLE" in state_name.upper():
                    self.slip_indicator.setStyleSheet("color: #f39c12")
                    self.slip_label.setStyleSheet("color: #f39c12")
                else:
                    self.slip_indicator.setStyleSheet("color: #2ecc71")
                    self.slip_label.setStyleSheet("")

    def reset(self):
        self.slip_label.setText("--")
        self.slip_indicator.setStyleSheet("color: #2ecc71")
        self.force_plots.reset()
