import faulthandler
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from app.main_window import MainWindow


def _enable_crash_dump():
    """段错误等原生崩溃时把所有线程堆栈写入 crash 日志，便于现场排查。

    打包版写到 exe 同级目录，源码运行写到项目根目录；
    文件打不开时退化到 stderr。
    """
    try:
        base = (
            Path(sys.executable).resolve().parent
            if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parent
        )
        stream = open(base / "vitai_viewer_crash.log", "a", encoding="utf-8")
        faulthandler.enable(file=stream, all_threads=True)
        _enable_crash_dump._stream = stream  # 保持文件句柄存活
    except OSError:
        faulthandler.enable(all_threads=True)


# 白色 + 浅灰色主题
APP_STYLESHEET = """
QMainWindow { background-color: #ffffff; }
QWidget { color: #2b2b2b; }
#DevicePanel { background-color: #f5f5f5; }
QGroupBox {
    background-color: #f7f7f7;
    border: 1px solid #dddddd;
    border-radius: 6px;
    margin-top: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #333333;
}
QFrame#SensorRow {
    background-color: #f7f7f7;
    border: 1px solid #dddddd;
    border-radius: 6px;
}
QComboBox, QPushButton, QSpinBox {
    background-color: #ffffff;
    border: 1px solid #c8c8c8;
    border-radius: 4px;
    padding: 3px 8px;
}
QComboBox:disabled, QPushButton:disabled, QSpinBox:disabled {
    color: #b0b0b0;
    background-color: #f2f2f2;
}
QPushButton:hover { background-color: #ececec; }
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #c8c8c8;
    selection-background-color: #e6e6e6;
    selection-color: #2b2b2b;
}
QScrollArea { border: none; background-color: #ffffff; }
QScrollBar:vertical {
    background: transparent;
    width: 12px;
    margin: 2px 2px 2px 0;
}
QScrollBar::handle:vertical {
    background: #c0c0c0;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #a0a0a0;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
    background: none;
    border: none;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
QMessageBox {
    background-color: #ffffff;
}
QMessageBox QLabel {
    color: #2b2b2b;
}
QStatusBar { background-color: #f5f5f5; color: #555555; }
"""


def main():
    _enable_crash_dump()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
