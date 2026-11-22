"""
Solver package - Samsung XR Track 3 Trajectory Estimators.
Implements Track A (A1, A2, A3) and Track B (B1, B2, B3).
"""

from .track_a import SolverA1, SolverA2, SolverA3
from .track_b import SolverB1, SolverB2, SolverB3

__all__ = [
    "SolverA1",
    "SolverA2",
    "SolverA3",
    "SolverB1",
    "SolverB2",
    "SolverB3",
]
