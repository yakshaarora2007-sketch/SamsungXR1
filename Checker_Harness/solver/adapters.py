"""
adapters.py - Adapters connecting Track A (A1, A2, A3) and Track B (B1, B2, B3)
to the Checker_Harness evaluation and benchmarking infrastructure.
"""

from typing import Dict, Optional, Any
import numpy as np
import pandas as pd

from .adapter import BaseSolverAdapter, SolverStatus, TrajectoryEstimate

# Import solvers from Solver package
import sys
from pathlib import Path
_workspace_dir = Path(__file__).resolve().parent.parent.parent
if str(_workspace_dir / "Solver") not in sys.path:
    sys.path.insert(0, str(_workspace_dir / "Solver"))
if str(_workspace_dir) not in sys.path:
    sys.path.insert(0, str(_workspace_dir))

from Solver.track_a import SolverA1, SolverA2, SolverA3
from Solver.track_b import SolverB1, SolverB2, SolverB3


def _to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Converts scalar-first [w, x, y, z] to scalar-last [x, y, z, w]."""
    return np.column_stack([q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]])


class A1Adapter(BaseSolverAdapter):
    """Adapter for Approach A1 (Classical Baseline)."""

    def __init__(self):
        self._solver = SolverA1()

    @property
    def name(self) -> str:
        return "A1"

    @property
    def track(self) -> str:
        return "Track A"

    @property
    def status(self) -> SolverStatus:
        return SolverStatus.COMPLETED

    def estimate_trajectory(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryEstimate:
        res = self._solver.solve(
            imu_df=imu_df,
            checkpoints_df=checkpoints_df,
            initial_state=initial_state,
        )
        return TrajectoryEstimate(
            timestamps=res["timestamps"],
            positions=res["positions"],
            orientations=_to_xyzw(res["orientations"]),
            metadata={
                "approach": "A1",
                "track": "Track A",
                "sample_count": len(res["timestamps"]),
            },
        )


class A2Adapter(BaseSolverAdapter):
    """Adapter for Approach A2 (A1 + Batch Gauss-Newton Polish)."""

    def __init__(self):
        self._solver = SolverA2()

    @property
    def name(self) -> str:
        return "A2"

    @property
    def track(self) -> str:
        return "Track A"

    @property
    def status(self) -> SolverStatus:
        return SolverStatus.COMPLETED

    def estimate_trajectory(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryEstimate:
        res = self._solver.solve(
            imu_df=imu_df,
            checkpoints_df=checkpoints_df,
            initial_state=initial_state,
        )
        return TrajectoryEstimate(
            timestamps=res["timestamps"],
            positions=res["positions"],
            orientations=_to_xyzw(res["orientations"]),
            metadata={
                "approach": "A2",
                "track": "Track A",
                "sample_count": len(res["timestamps"]),
            },
        )


class A3Adapter(BaseSolverAdapter):
    """Adapter for Approach A3 (A2 + Robust-Statistics Upgrades)."""

    def __init__(self):
        self._solver = SolverA3()

    @property
    def name(self) -> str:
        return "A3"

    @property
    def track(self) -> str:
        return "Track A"

    @property
    def status(self) -> SolverStatus:
        return SolverStatus.COMPLETED

    def estimate_trajectory(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryEstimate:
        res = self._solver.solve(
            imu_df=imu_df,
            checkpoints_df=checkpoints_df,
            initial_state=initial_state,
        )
        return TrajectoryEstimate(
            timestamps=res["timestamps"],
            positions=res["positions"],
            orientations=_to_xyzw(res["orientations"]),
            metadata={
                "approach": "A3",
                "track": "Track A",
                "sample_count": len(res["timestamps"]),
            },
        )


class B1Adapter(BaseSolverAdapter):
    """Adapter for Approach B1 (Track A + BiTCN C=32, 6 Blocks)."""

    def __init__(self, drop_nn: bool = False):
        self._solver = SolverB1(drop_nn=drop_nn)

    @property
    def name(self) -> str:
        return "B1"

    @property
    def track(self) -> str:
        return "Track B"

    @property
    def status(self) -> SolverStatus:
        return SolverStatus.COMPLETED

    def estimate_trajectory(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryEstimate:
        res = self._solver.solve(
            imu_df=imu_df,
            checkpoints_df=checkpoints_df,
            initial_state=initial_state,
        )
        return TrajectoryEstimate(
            timestamps=res["timestamps"],
            positions=res["positions"],
            orientations=_to_xyzw(res["orientations"]),
            metadata={
                "approach": "B1",
                "track": "Track B",
                "nn_applied": res.get("nn_applied", True),
                "sample_count": len(res["timestamps"]),
            },
        )


class B2Adapter(BaseSolverAdapter):
    """Adapter for Approach B2 (Track A + BiTCN C=48, 6 Blocks)."""

    def __init__(self, drop_nn: bool = False):
        self._solver = SolverB2(drop_nn=drop_nn)

    @property
    def name(self) -> str:
        return "B2"

    @property
    def track(self) -> str:
        return "Track B"

    @property
    def status(self) -> SolverStatus:
        return SolverStatus.COMPLETED

    def estimate_trajectory(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryEstimate:
        res = self._solver.solve(
            imu_df=imu_df,
            checkpoints_df=checkpoints_df,
            initial_state=initial_state,
        )
        return TrajectoryEstimate(
            timestamps=res["timestamps"],
            positions=res["positions"],
            orientations=_to_xyzw(res["orientations"]),
            metadata={
                "approach": "B2",
                "track": "Track B",
                "nn_applied": res.get("nn_applied", True),
                "sample_count": len(res["timestamps"]),
            },
        )


class B3Adapter(BaseSolverAdapter):
    """Adapter for Approach B3 (Track A + BiTCN C=64, 8 Blocks Chunked)."""

    def __init__(self, drop_nn: bool = False):
        self._solver = SolverB3(drop_nn=drop_nn)

    @property
    def name(self) -> str:
        return "B3"

    @property
    def track(self) -> str:
        return "Track B"

    @property
    def status(self) -> SolverStatus:
        return SolverStatus.COMPLETED

    def estimate_trajectory(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryEstimate:
        res = self._solver.solve(
            imu_df=imu_df,
            checkpoints_df=checkpoints_df,
            initial_state=initial_state,
        )
        return TrajectoryEstimate(
            timestamps=res["timestamps"],
            positions=res["positions"],
            orientations=_to_xyzw(res["orientations"]),
            metadata={
                "approach": "B3",
                "track": "Track B",
                "nn_applied": res.get("nn_applied", True),
                "sample_count": len(res["timestamps"]),
            },
        )


def get_solver_adapter(approach_name: str) -> BaseSolverAdapter:
    """Factory retrieving solver adapter by canonical approach name."""
    mapping = {
        "a1": A1Adapter,
        "a2": A2Adapter,
        "a3": A3Adapter,
        "b1": B1Adapter,
        "b2": B2Adapter,
        "b3": B3Adapter,
    }
    key = approach_name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown approach: {approach_name}. Available approaches: A1, A2, A3, B1, B2, B3.")
    return mapping[key]()
