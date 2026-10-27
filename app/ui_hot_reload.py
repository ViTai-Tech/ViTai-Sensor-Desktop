"""UI 就地热更新：重载样式 / widgets，不重启进程、不断开传感器。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

STYLES_PATH = Path(__file__).resolve().parent / "styles.qss"

# 依赖顺序：先叶子模块
_WIDGET_MODULES = (
    "app.widgets.checkable_combo",
    "app.widgets.image_viewer",
    "app.widgets.data_panel",
    "app.widgets.device_panel",
    "app.widgets.multi_sensor_viewer",
)


def is_ui_only_change(paths: list[Path], root: Path) -> bool:
    """变更文件是否全部属于 UI（widgets / styles）。"""
    if not paths:
        return False
    for path in paths:
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return False
        if rel == "app/styles.qss":
            continue
        if rel.startswith("app/widgets/") and rel.endswith(".py"):
            continue
        return False
    return True


def is_styles_only_change(paths: list[Path], root: Path) -> bool:
    if not paths:
        return False
    for path in paths:
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return False
        if rel != "app/styles.qss":
            return False
    return True


def load_stylesheet() -> str:
    text = STYLES_PATH.read_text(encoding="utf-8")
    # Qt QSS 的 url() 需要绝对路径（且用正斜杠），否则打包/cwd 变化后箭头图会丢
    arrow = (STYLES_PATH.parent / "icons" / "combo_down_arrow.png").resolve()
    arrow_url = arrow.as_posix()
    return text.replace(
        "url(app/icons/combo_down_arrow.png)",
        f"url({arrow_url})",
    )


def reload_widget_modules() -> None:
    for name in _WIDGET_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            importlib.import_module(name)
        else:
            importlib.reload(mod)
