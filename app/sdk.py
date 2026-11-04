"""Shared imports for the original and white-label SDKs."""
from importlib import import_module
from importlib.util import find_spec

import sys

if getattr(sys, "frozen", False):
    # Packaged applications must use the SDK selected during this build.
    from app.build_info import SDK_MODULE
else:
    SDK_MODULE = "bcvts_gel" if find_spec("bcvts_gel") is not None else "pyvitaisdk"
_sdk = import_module(SDK_MODULE)

VTSensor = _sdk.VTSensor
VTSensorType = _sdk.VTSensorType
VTSDataType = _sdk.VTSDataType
VTSError = _sdk.VTSError
VTSDeviceFinder = _sdk.VTSDeviceFinder

try:
    _exceptions = import_module(f"{SDK_MODULE}.common.exceptions")
except ModuleNotFoundError as exc:
    if exc.name not in (
        f"{SDK_MODULE}.common",
        f"{SDK_MODULE}.common.exceptions",
    ):
        raise
    FrameNotAvailableError = None
else:
    FrameNotAvailableError = getattr(_exceptions, "FrameNotAvailableError", None)
