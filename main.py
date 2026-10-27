import os
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import QTimer

from app.main_window import MainWindow
from app.ui_hot_reload import (
    is_styles_only_change,
    is_ui_only_change,
    load_stylesheet,
)

# 与 dev.py 约定：热更新请求退出码
_RELOAD_EXIT_CODE = 100
_ENV_RELOAD = "VITAI_DEV_RELOAD"
_IGNORE_DIRS = {
    ".git",
    ".history",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    "wheels",
}


def _apply_dark_palette(app: QApplication):
    palette = QPalette()
    bg = QColor("#0a0f1a")
    panel = QColor("#0f172a")
    text = QColor("#e2e8f0")
    muted = QColor("#94a3b8")
    accent = QColor("#0e7490")
    highlight_text = QColor("#ffffff")

    palette.setColor(QPalette.ColorRole.Window, bg)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, panel)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#121a2b"))
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, QColor("#152238"))
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#67e8f9"))
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, highlight_text)
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#121a2b"))
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.PlaceholderText, muted)
    palette.setColor(QPalette.ColorRole.Link, QColor("#22d3ee"))
    app.setPalette(palette)


def _iter_watch_files(root: Path):
    main_py = root / "main.py"
    if main_py.is_file():
        yield main_py

    styles = root / "app" / "styles.qss"
    if styles.is_file():
        yield styles

    for path in root.rglob("*.py"):
        if path.resolve() == main_py.resolve():
            continue
        if any(part in _IGNORE_DIRS for part in path.parts):
            continue
        yield path


def _file_snapshot(root: Path) -> dict[Path, int]:
    result: dict[Path, int] = {}
    for path in _iter_watch_files(root):
        try:
            result[path] = path.stat().st_mtime_ns
        except OSError:
            continue
    return result


def _changed_paths(
    old: dict[Path, int],
    new: dict[Path, int],
) -> list[Path]:
    keys = old.keys() | new.keys()
    return sorted(
        (p for p in keys if old.get(p) != new.get(p)),
        key=lambda p: str(p),
    )


def _install_hot_reload(app: QApplication, window: MainWindow) -> None:
    """开发模式热更新：

    - app/styles.qss / app/widgets/** → UI 就地刷新（不断传感器）
    - 其它源码 → 优雅退出，由 dev.py 整进程重启
    """
    root = Path(__file__).resolve().parent
    state = {
        "mtimes": _file_snapshot(root),
        "busy": False,
    }

    def _apply_stylesheet():
        try:
            app.setStyleSheet(load_stylesheet())
            print("[reload] 样式已热更新")
            window.status_bar.showMessage("样式已热更新", 1500)
        except OSError as exc:
            print(f"[reload] 读取样式失败: {exc}")

    def _request_process_reload():
        if state["busy"]:
            return
        state["busy"] = True
        print("[reload] 非 UI 变更，正在优雅退出以便整进程重启…")
        for widget in list(app.topLevelWidgets()):
            widget.close()
        app.exit(_RELOAD_EXIT_CODE)

    def _request_ui_reload(styles_only: bool):
        if state["busy"]:
            return
        state["busy"] = True
        try:
            if styles_only:
                _apply_stylesheet()
            else:
                _apply_stylesheet()
                window.reload_ui()
        except Exception as exc:
            print(f"[reload] UI 热更新失败: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            state["busy"] = False
            state["mtimes"] = _file_snapshot(root)

    def _on_change(changed: list[Path]):
        rels = []
        for p in changed:
            try:
                rels.append(str(p.relative_to(root)))
            except ValueError:
                rels.append(str(p))
        print(f"[reload] 检测到变更: {', '.join(rels)}")

        if is_styles_only_change(changed, root):
            _request_ui_reload(styles_only=True)
            return
        if is_ui_only_change(changed, root):
            _request_ui_reload(styles_only=False)
            return
        _request_process_reload()

    def _poll():
        if state["busy"]:
            return
        current = _file_snapshot(root)
        if current == state["mtimes"]:
            return
        changed = _changed_paths(state["mtimes"], current)
        state["mtimes"] = current
        # 防抖：等编辑器写完
        QTimer.singleShot(500, lambda: _on_change(changed))

    timer = QTimer(app)
    timer.setInterval(400)
    timer.timeout.connect(_poll)
    timer.start()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    _apply_dark_palette(app)

    font = QFont("Segoe UI" if sys.platform == "win32" else "Noto Sans", 10)
    app.setFont(font)
    app.setStyleSheet(load_stylesheet())

    window = MainWindow()
    window.show()

    if os.environ.get(_ENV_RELOAD) == "1":
        _install_hot_reload(app, window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
