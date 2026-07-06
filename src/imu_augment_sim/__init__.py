"""opensim_imu_augmentation.

This package provides utilities for generating synthetic IMU trajectories from
biomechanical simulations and analysing the results.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version handling
# ---------------------------------------------------------------------------
from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version(__name__)
except PackageNotFoundError:  # package not installed (e.g., source tree)
    __version__ = "0.0.0+dirty"

# ---------------------------------------------------------------------------
# Public API re‑exports
# ---------------------------------------------------------------------------
from .dataset import ExerciseDataset

__all__: list[str] = [
    "__version__",
]

