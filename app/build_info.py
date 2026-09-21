# app/build_info.py
"""
构建信息。

该文件中的值会在执行 build.py 时自动更新。
运行时 UI 只读取这里的信息，不直接执行 Git 命令。
"""

GIT_BRANCH = "unknown"
SDK_WHEEL = "unknown"


def get_build_info() -> str:
    """返回适合显示在 UI 中的构建信息。"""
    return f"Git: {GIT_BRANCH} | SDK: {SDK_WHEEL}"


def get_window_title(base_title: str = "ViTai 视触觉传感器查看器") -> str:
    """生成主窗口标题。"""
    return f"{base_title} | {get_build_info()}"