#!/usr/bin/env python3
"""Build script for ViTaiViewer - auto-detects Windows/Linux."""

import os
import sys
import platform


def get_wheel_path():
    """Return the appropriate pyvitaisdk wheel for the current platform."""
    base = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == "win32":
        wheel_dir = os.path.join(base, "wheels", "windows")
    else:
        wheel_dir = os.path.join(base, "wheels", "linux")

    wheels = [f for f in os.listdir(wheel_dir) if f.endswith(".whl")]
    if not wheels:
        raise FileNotFoundError(f"No .whl found in {wheel_dir}")
    return os.path.join(wheel_dir, wheels[0])


def install_sdk():
    """Install pyvitaisdk from the platform-specific wheel."""
    wheel = get_wheel_path()
    print(f"[1/4] Installing pyvitaisdk from: {os.path.basename(wheel)}")
    ret = os.system(f'"{sys.executable}" -m pip install "{wheel}"')
    if ret != 0:
        print("WARNING: pyvitaisdk install failed. Ensure the wheel is present.")
    else:
        print("       OK")


def install_deps():
    """Install cross-platform Python dependencies."""
    print("[2/4] Installing dependencies...")
    ret = os.system(
        f'"{sys.executable}" -m pip install '
        "PyQt6 pyqtgraph opencv-python numpy"
    )
    if ret != 0:
        raise RuntimeError("Failed to install dependencies.")


def build_exe():
    """Build the single-file executable with PyInstaller."""
    sep = ";" if sys.platform == "win32" else ":"
    name = "ViTaiViewer"

    cmd = (
        f'pyinstaller --onefile --windowed --name "{name}" '
        f'--add-data "app{sep}app" '
        f"--collect-all pyvitaisdk "
        f"--collect-submodules scipy "
        f"--collect-submodules skimage "
        f"--collect-submodules cryptography "
        f"--collect-submodules appdirs "
        f"--collect-all comtypes "
        f"--hidden-import PyQt6 "
        f"--hidden-import pyqtgraph "
        f"--hidden-import cv2 "
        f"--hidden-import colorlog "
        f"--hidden-import coloredlogs "
        f"--hidden-import onnxruntime "
        f"--hidden-import matplotlib "
        f"--hidden-import appdirs "
        f"main.py"
    )

    print(f"[3/4] Building {name}...")
    print(f"      Platform: {platform.system()} {platform.machine()}")
    print(f"      Python: {platform.python_version()}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError("Build failed.")


def main():
    print("=" * 50)
    print("  ViTai Sensor Viewer - Build Script")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print("=" * 50)
    print()

    install_sdk()
    print()
    install_deps()
    print()
    build_exe()

    dist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
    print()
    print("[4/4] Build complete!")
    exe_name = "ViTaiViewer.exe" if sys.platform == "win32" else "ViTaiViewer"
    print(f"      Output: {os.path.join(dist, exe_name)}")


if __name__ == "__main__":
    main()
