"""
solver package initialization.
Exports BaseSolverAdapter, TrajectoryEstimate, SolverStatus, and approach adapters.
"""

from .adapter import BaseSolverAdapter, TrajectoryEstimate, SolverStatus, PendingSolverAdapter
from .adapters import (
    A1Adapter,
    A2Adapter,
    A3Adapter,
    B1Adapter,
    B2Adapter,
    B3Adapter,
    get_solver_adapter,
)

__all__ = [
    "BaseSolverAdapter",
    "TrajectoryEstimate",
    "SolverStatus",
    "PendingSolverAdapter",
    "A1Adapter",
    "A2Adapter",
    "A3Adapter",
    "B1Adapter",
    "B2Adapter",
    "B3Adapter",
    "get_solver_adapter",
]
