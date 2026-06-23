import sys

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
import cv2
import numpy as np
from pyvitaisdk import VTSDataType


_MONO_FONT = "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 120)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pixmap = None
        self._source_array = None
        self._loading = False
        self.setStyleSheet("background-color: #1a1a1a; border-radius: 6px;")
        self.setText("无信号")

    def set_loading(self, loading):
        self._loading = loading
        if loading:
            self.setPixmap(QPixmap())
            self.setText("加载中...")
            self.setFont(QFont("", 20, QFont.Weight.Bold))
            self.setStyleSheet("background-color: #1a1a1a; border-radius: 6px; color: #ffffff;")
        else:
            self.setText("无信号")
            self.setFont(QFont())
            self.setStyleSheet("background-color: #1a1a1a; border-radius: 6px;")

    def set_image(self, img_array, is_depth=False):
        if img_array is None:
            return

        if len(img_array.shape) == 2:
            img = img_array.copy()
            valid = ~np.isnan(img) & ~np.isinf(img)
            if valid.any():
                if is_depth:
                    vmin, vmax = 0.0, 1.5
                else:
                    vmin, vmax = np.percentile(img[valid], [2, 98])
                img = np.clip(img, vmin, vmax)
                img = (img - vmin) / (vmax - vmin + 1e-8) * 255
            else:
                img = np.zeros_like(img)
            img = img.astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qt_img)
        self._source_array = rgb
        self._update_display()

    def _update_display(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
        else:
            self.setPixmap(QPixmap())
            self.setText("无信号")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class ForceValueWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(18)

        val_font = QFont(_MONO_FONT, 28, QFont.Weight.Bold)
        lbl_font = QFont("", 14, QFont.Weight.Bold)

        fx_row = QHBoxLayout()
        fx_row.setSpacing(8)
        self.fx_val = QLabel("0.000")
        self.fx_val.setFont(val_font)
        self.fx_val.setStyleSheet("color: #e74c3c")
        fx_row.addStretch()
        fx_row.addWidget(QLabel("Fx", font=lbl_font))
        fx_row.addWidget(self.fx_val)
        fx_row.addWidget(QLabel("N"))
        fx_row.addStretch()
        layout.addLayout(fx_row)
        layout.addStretch(1)

        fy_row = QHBoxLayout()
        fy_row.setSpacing(8)
        self.fy_val = QLabel("0.000")
        self.fy_val.setFont(val_font)
        self.fy_val.setStyleSheet("color: #2ecc71")
        fy_row.addStretch()
        fy_row.addWidget(QLabel("Fy", font=lbl_font))
        fy_row.addWidget(self.fy_val)
        fy_row.addWidget(QLabel("N"))
        fy_row.addStretch()
        layout.addLayout(fy_row)
        layout.addStretch(1)

        fz_row = QHBoxLayout()
        fz_row.setSpacing(8)
        self.fz_val = QLabel("0.000")
        self.fz_val.setFont(val_font)
        self.fz_val.setStyleSheet("color: #3498db")
        fz_row.addStretch()
        fz_row.addWidget(QLabel("Fz", font=lbl_font))
        fz_row.addWidget(self.fz_val)
        fz_row.addWidget(QLabel("N"))
        fz_row.addStretch()
        layout.addLayout(fz_row)
        layout.addStretch(1)

    def update_values(self, f):
        if f is not None and len(f) >= 3:
            self.fx_val.setText(f"{f[0]:.3f}")
            self.fy_val.setText(f"{f[1]:.3f}")
            self.fz_val.setText(f"{f[2]:.3f}")


class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._current_data = {}

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        grid = QGridLayout()
        grid.setSpacing(2)

        self.warped_view = ImageLabel()
        self.depth_view = ImageLabel()
        self.marker_view = ImageLabel()
        self.force_values = ForceValueWidget()

        grid.addWidget(QLabel("原始图"), 0, 0, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QLabel(""), 0, 1)
        grid.addWidget(self.warped_view, 1, 0)
        grid.addWidget(self.force_values, 1, 1)
        grid.addWidget(QLabel("深度图"), 2, 0, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(QLabel("标记点"), 2, 1, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.depth_view, 3, 0)
        grid.addWidget(self.marker_view, 3, 1)

        grid.setRowStretch(0, 0)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 0)
        grid.setRowStretch(3, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        layout.addLayout(grid)

    def set_loading(self, loading):
        self.warped_view.set_loading(loading)
        self.depth_view.set_loading(loading)
        self.marker_view.set_loading(loading)

    def update_data(self, data: dict):
        if VTSDataType.WARPED_IMG in data:
            self.warped_view.set_image(data[VTSDataType.WARPED_IMG])
        if VTSDataType.DEPTH_MAP in data:
            self.depth_view.set_image(data[VTSDataType.DEPTH_MAP], is_depth=True)
        if VTSDataType.MARKER_IMG in data:
            self.marker_view.set_image(data[VTSDataType.MARKER_IMG])
        if VTSDataType.FORCE6D_VECTOR in data:
            self.force_values.update_values(data[VTSDataType.FORCE6D_VECTOR])
