import sys

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QFrame,
    QScrollArea,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from app.widgets.image_viewer import ImageLabel
from app.widgets.data_panel import ForcePlots
from pyvitaisdk import VTSDataType


_MONO_FONT = "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"


class SensorRowWidget(QFrame):
    """单个传感器的显示行：名称 + 原图(力叠加在左上角) + 深度图 + 标记点 + 折线图，全部在同一行。"""

    def __init__(self, sn, parent=None):
        super().__init__(parent)
        self.sn = sn
        self.setObjectName("SensorRow")
        self._setup_ui()

    def _setup_ui(self):
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 4, 6, 4)
        h.setSpacing(6)

        # 传感器名称（行首）
        name = QLabel(self.sn)
        name.setFixedWidth(96)
        name.setWordWrap(True)
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setStyleSheet("font-weight: bold; color: #333; font-size: 12px;")
        h.addWidget(name)

        # 原图 + Fx/Fy/Fz（叠加在原图左上角）
        self.raw_view = ImageLabel()
        self.force_label = QLabel("Fx --  Fy --  Fz --")
        self.force_label.setTextFormat(Qt.TextFormat.RichText)
        self.force_label.setFont(QFont(_MONO_FONT, 8, QFont.Weight.Bold))
        self.force_label.setStyleSheet(
            "background-color: rgba(255,255,255,0.85); padding: 1px 3px; border-radius: 2px;"
        )
        raw_tile = self._tile("原图", self.raw_view, overlay=self.force_label)

        # 深度图
        self.depth_view = ImageLabel()
        depth_tile = self._tile("深度图", self.depth_view)

        # 标记点
        self.marker_view = ImageLabel()
        marker_tile = self._tile("标记点", self.marker_view)

        # 折线图（三根独立，竖排一列）+ 滑动状态（右下角）
        self.force_plots = ForcePlots()
        for p in (self.force_plots.fx_plot, self.force_plots.fy_plot, self.force_plots.fz_plot):
            p.setFixedHeight(34)
        self.slip_label = QLabel("● --")
        self.slip_label.setFont(QFont("", 10, QFont.Weight.Bold))
        self.slip_label.setStyleSheet("color: #2ecc71;")

        chart_col = QWidget()
        cv = QVBoxLayout(chart_col)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(2)
        cv.addWidget(self.force_plots)
        cv.addStretch(1)
        slip_row = QHBoxLayout()
        slip_row.addStretch()
        slip_row.addWidget(self.slip_label)
        cv.addLayout(slip_row)

        h.addWidget(raw_tile, 3)
        h.addWidget(depth_tile, 3)
        h.addWidget(marker_tile, 3)
        h.addWidget(chart_col, 4)

    def _tile(self, title, view, overlay=None):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(2)
        cap = QLabel(title)
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.addWidget(cap)
        if overlay is not None:
            body = QWidget()
            g = QGridLayout(body)
            g.setContentsMargins(0, 0, 0, 0)
            g.setSpacing(0)
            g.addWidget(view, 0, 0)
            g.addWidget(overlay, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            # 让图像填满 body，而不是被 Ignored 尺寸策略压到最小尺寸
            g.setColumnStretch(0, 1)
            g.setRowStretch(0, 1)
            l.addWidget(body, 1)
        else:
            l.addWidget(view, 1)
        return w

    def set_loading(self, loading):
        for view in (self.raw_view, self.depth_view, self.marker_view):
            view.set_loading(loading)

    def update_data(self, data):
        if VTSDataType.WARPED_IMG in data:
            self.raw_view.set_image(data[VTSDataType.WARPED_IMG])
        if VTSDataType.DEPTH_MAP in data:
            self.depth_view.set_image(data[VTSDataType.DEPTH_MAP], is_depth=True)
        if VTSDataType.MARKER_IMG in data:
            self.marker_view.set_image(data[VTSDataType.MARKER_IMG])
        if VTSDataType.FORCE6D_VECTOR in data:
            f = data[VTSDataType.FORCE6D_VECTOR]
            self._set_force(f)
            if f is not None and len(f) >= 6:
                self.force_plots.update_data(f)
        if VTSDataType.SLIP_STATE in data:
            self._set_slip(data[VTSDataType.SLIP_STATE])

    def _set_force(self, f):
        if f is None or len(f) < 3:
            return
        self.force_label.setText(
            f'<span style="color:#e74c3c;">Fx {f[0]:.2f}</span> '
            f'<span style="color:#2ecc71;">Fy {f[1]:.2f}</span> '
            f'<span style="color:#3498db;">Fz {f[2]:.2f}</span>'
        )

    def _set_slip(self, state):
        if state is None:
            return
        name = state.name if hasattr(state, "name") else str(state)
        if "SLIP" in name.upper():
            self.slip_label.setStyleSheet("color: #e74c3c;")
        else:
            self.slip_label.setStyleSheet("color: #2ecc71;")
        self.slip_label.setText(f"● {name}")


class MultiSensorViewer(QWidget):
    """按行显示多个传感器内容的滚动区域。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(8)

        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll)

    def set_sensors(self, sns):
        self._clear_rows()
        for sn in sns:
            row = SensorRowWidget(sn)
            self._rows[sn] = row
            self._layout.addWidget(row, 1)

    def remove_sensor(self, sn):
        row = self._rows.pop(sn, None)
        if row is not None:
            self._layout.removeWidget(row)
            row.deleteLater()

    def _clear_rows(self):
        while self._layout.count() > 0:
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._rows.clear()

    def update_data(self, sn, data):
        row = self._rows.get(sn)
        if row is not None:
            row.update_data(data)

    def set_loading(self, sns, loading):
        for sn in sns:
            row = self._rows.get(sn)
            if row is not None:
                row.set_loading(loading)

    def reset(self):
        self._clear_rows()

    def count(self):
        return len(self._rows)
