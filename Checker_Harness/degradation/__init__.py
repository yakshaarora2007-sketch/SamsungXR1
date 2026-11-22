"""
degradation package initialization.
Re-exports Degrader's degradation functions to prevent name collision with Degrader/degradation.py.
"""

import importlib.util
from pathlib import Path
import sys

_possible_degrader_paths = [
    Path(__file__).resolve().parent.parent.parent / "Degrader",
    Path("Degrader").resolve(),
    Path("../Degrader").resolve(),
    Path("e:/SamsungXR/Degrader"),
]

for p in _possible_degrader_paths:
    deg_file = p / "degradation.py"
    if deg_file.exists():
        if str(p.resolve()) not in sys.path:
            sys.path.insert(0, str(p.resolve()))
        spec = importlib.util.spec_from_file_location("_orig_degradation", str(deg_file))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        apply_sensor_degradation = mod.apply_sensor_degradation
        identify_imu_columns = mod.identify_imu_columns
        DEFAULT_DEGRADATION_CONFIG = mod.DEFAULT_DEGRADATION_CONFIG
        break

from .adapter import DegradationAdapter

__all__ = [
    "DegradationAdapter",
    "apply_sensor_degradation",
    "identify_imu_columns",
    "DEFAULT_DEGRADATION_CONFIG",
]
