"""
solver/adapter.py - Modular trajectory solver interface and pending adapter.
Clearly marks solver evaluation as pending without generating dummy trajectories.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from typing import Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd

try:
    from memory.monitor import AlgorithmMemoryMonitor, MemoryProfile, MemoryConstraintStatus
except ImportError:
    from Checker_Harness.memory.monitor import AlgorithmMemoryMonitor, MemoryProfile, MemoryConstraintStatus


class SolverStatus(Enum):
    PENDING = "Pending solver implementation"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"


@dataclass
class TrajectoryEstimate:
    """Estimated trajectory produced by a trajectory solver."""
    timestamps: np.ndarray  # int64 nanoseconds (N,)
    positions: np.ndarray   # float64 [x, y, z] (N, 3)
    orientations: np.ndarray  # float64 quaternions [x, y, z, w] or rotation matrices (N, 4)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSolverAdapter(ABC):
    """Abstract interface for trajectory estimation solvers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name or identifier of the solver."""
        pass

    @property
    @abstractmethod
    def status(self) -> SolverStatus:
        """Current status of the solver integration."""
        pass

    @abstractmethod
    def estimate_trajectory(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryEstimate:
        """
        Estimates the 6-DOF trajectory given IMU inputs and optional checkpoints/priors.
        Must be implemented by concrete solver implementations.
        """
        pass

    def estimate_with_memory_monitoring(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ) -> Tuple[TrajectoryEstimate, MemoryProfile]:
        """
        Executes estimate_trajectory while measuring algorithm-level memory footprint
        and enforcing the <= 8 MB constraint.
        """
        monitor = AlgorithmMemoryMonitor(operation_name=f"Solver_{self.name}", strict=strict)
        monitor.start()
        monitor.track_buffer("solver_imu_input", imu_df)
        if checkpoints_df is not None:
            monitor.track_buffer("solver_checkpoints", checkpoints_df)

        try:
            t0_solve = time.perf_counter()
            trajectory = self.estimate_trajectory(
                imu_df=imu_df,
                checkpoints_df=checkpoints_df,
                initial_state=initial_state,
            )
            solve_time = time.perf_counter() - t0_solve
            trajectory.metadata["solve_runtime_s"] = solve_time
            monitor.track_buffer("estimated_positions", trajectory.positions)
            monitor.track_buffer("estimated_orientations", trajectory.orientations)
            return trajectory, monitor.stop()
        except Exception:
            monitor.stop()
            raise


class PendingSolverAdapter(BaseSolverAdapter):
    """
    Placeholder adapter for future solver integration.
    Explicitly marks evaluation as pending without generating dummy data.
    """

    @property
    def name(self) -> str:
        return "PendingSolver"

    @property
    def status(self) -> SolverStatus:
        return SolverStatus.PENDING

    def estimate_trajectory(
        self,
        imu_df: pd.DataFrame,
        checkpoints_df: Optional[pd.DataFrame] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryEstimate:
        raise NotImplementedError(
            "Trajectory solver has not been implemented yet. "
            "Solver evaluation is pending implementation."
        )
