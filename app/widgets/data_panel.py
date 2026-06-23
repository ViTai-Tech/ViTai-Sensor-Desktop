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
        self._buffer = deque(maxlen=200)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        pg.setConfigOptions(antialias=True)
        bg_color = self.palette().color(self.backgroundRole()).name()

        self.fx_plot = pg.PlotWidget(title="Fx (N)")
        self.fx_plot.setBackground(bg_color)
        self.fx_plot.setLabel("left", "N")
        self.fx_plot.setLabel("bottom", "")
        self.fx_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fx_curve = self.fx_plot.plot(pen=pg.mkPen("#e74c3c", width=2))

        self.fy_plot = pg.PlotWidget(title="Fy (N)")
        self.fy_plot.setBackground(bg_color)
        self.fy_plot.setLabel("left", "N")
        self.fy_plot.setLabel("bottom", "")
        self.fy_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fy_curve = self.fy_plot.plot(pen=pg.mkPen("#2ecc71", width=2))

        self.fz_plot = pg.PlotWidget(title="Fz (N)")
        self.fz_plot.setBackground(bg_color)
        self.fz_plot.setLabel("left", "N")
        self.fz_plot.setLabel("bottom", "")
        self.fz_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fz_curve = self.fz_plot.plot(pen=pg.mkPen("#3498db", width=2))

        layout.addWidget(self.fx_plot)
        layout.addWidget(self.fy_plot)
        layout.addWidget(self.fz_plot)

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

    def reset(self):
        self._buffer.clear()
        x = []
        y = []
        self.fx_curve.setData(x, y)
        self.fy_curve.setData(x, y)
        self.fz_curve.setData(x, y)


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
