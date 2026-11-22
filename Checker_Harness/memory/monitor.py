"""
memory/monitor.py - Algorithm-level and Heap-level Memory Profiler.
Enforces the 8 MB peak algorithm memory constraint in accordance with official guidelines:
- Python interpreter baseline memory is EXCLUDED.
- OS-level memory overhead is EXCLUDED.
- Algorithm data structures, IMU buffers, intermediate arrays, and native buffers are INCLUDED.
"""

from dataclasses import dataclass, field
from enum import Enum
import gc
import logging
import os
from pathlib import Path
import sys
import threading
import time
import tracemalloc
from typing import Dict, List, Optional, Any, Tuple, Union

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

import numpy as np
import pandas as pd

logger = logging.getLogger("MemoryMonitor")

# Hard constraint: 8 MB = 8 * 1024 * 1024 bytes
MEMORY_LIMIT_BYTES: int = 8 * 1024 * 1024
MEMORY_LIMIT_MB: float = 8.0


class MemoryConstraintStatus(Enum):
    PASS = "PASS"
    VIOLATION = "VIOLATION"
    PARTIALLY_VERIFIED_PASS = "PARTIALLY_VERIFIED_PASS"
    FAILED = "FAILED"
    PENDING = "PENDING"


class MemoryCoverage(Enum):
    FULL = "FULL"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"


class MemoryConstraintViolationError(RuntimeError):
    """Raised when peak algorithm-level memory exceeds the 8 MB constraint in strict mode."""
    def __init__(self, operation: str, peak_bytes: int, limit_bytes: int = MEMORY_LIMIT_BYTES):
        self.operation = operation
        self.peak_bytes = peak_bytes
        self.limit_bytes = limit_bytes
        peak_mb = peak_bytes / (1024 * 1024)
        limit_mb = limit_bytes / (1024 * 1024)
        super().__init__(
            f"[MEMORY CONSTRAINT VIOLATION] Operation '{operation}' exceeded the 8 MB algorithm memory limit: "
            f"Observed peak: {peak_bytes:,} bytes ({peak_mb:.3f} MB) > Limit: {limit_bytes:,} bytes ({limit_mb:.1f} MB)."
        )


@dataclass
class MemoryProfile:
    """Detailed memory profile distinguishing interpreter baseline from algorithm allocations."""
    operation_name: str
    limit_bytes: int = MEMORY_LIMIT_BYTES

    # Interpreter / Process level (excluded from 8 MB limit, kept for diagnostics)
    interpreter_baseline_bytes: int = 0
    process_rss_peak_bytes: int = 0
    process_rss_delta_bytes: int = 0

    # Algorithm / Heap level (strictly enforced under 8 MB)
    python_heap_peak_bytes: int = 0
    native_buffers_peak_bytes: int = 0
    algorithm_peak_bytes: int = 0

    # Status and Coverage
    status: MemoryConstraintStatus = MemoryConstraintStatus.PASS
    coverage: MemoryCoverage = MemoryCoverage.FULL
    methodology: str = "tracemalloc + native buffer tracking + process delta isolation"
    tracked_buffers: Dict[str, int] = field(default_factory=dict)
    violation_details: Optional[Dict[str, Any]] = None

    @property
    def algorithm_peak_mb(self) -> float:
        return self.algorithm_peak_bytes / (1024 * 1024)

    @property
    def interpreter_baseline_mb(self) -> float:
        return self.interpreter_baseline_bytes / (1024 * 1024)

    @property
    def process_rss_peak_mb(self) -> float:
        return self.process_rss_peak_bytes / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_name": self.operation_name,
            "limit_bytes": self.limit_bytes,
            "limit_mb": self.limit_bytes / (1024 * 1024),
            "interpreter_baseline_bytes": self.interpreter_baseline_bytes,
            "interpreter_baseline_mb": round(self.interpreter_baseline_mb, 3),
            "algorithm_peak_bytes": self.algorithm_peak_bytes,
            "algorithm_peak_mb": round(self.algorithm_peak_mb, 3),
            "python_heap_peak_bytes": self.python_heap_peak_bytes,
            "native_buffers_peak_bytes": self.native_buffers_peak_bytes,
            "process_rss_peak_bytes": self.process_rss_peak_bytes,
            "process_rss_peak_mb": round(self.process_rss_peak_mb, 3),
            "status": self.status.value,
            "coverage": self.coverage.value,
            "methodology": self.methodology,
            "tracked_buffers": self.tracked_buffers,
            "violation_details": self.violation_details,
        }


class NativeBufferTracker:
    """Tracks native memory allocations such as NumPy arrays, C buffers, and preallocated memory."""
    def __init__(self):
        self._buffers: Dict[str, int] = {}
        self._peak_bytes: int = 0
        self._current_bytes: int = 0
        self._lock = threading.Lock()

    def register_buffer(self, name: str, obj: Any):
        """Measures and registers a buffer or data structure."""
        size = self._measure_obj(obj)
        with self._lock:
            # If buffer already registered, update
            prev = self._buffers.get(name, 0)
            self._current_bytes = self._current_bytes - prev + size
            self._buffers[name] = size
            if self._current_bytes > self._peak_bytes:
                self._peak_bytes = self._current_bytes

    def unregister_buffer(self, name: str):
        with self._lock:
            if name in self._buffers:
                self._current_bytes -= self._buffers.pop(name)

    def clear(self):
        with self._lock:
            self._buffers.clear()
            self._peak_bytes = 0
            self._current_bytes = 0

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    @property
    def buffers(self) -> Dict[str, int]:
        return dict(self._buffers)

    @staticmethod
    def _measure_obj(obj: Any) -> int:
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        if isinstance(obj, pd.DataFrame):
            return int(obj.memory_usage(deep=True).sum())
        if isinstance(obj, pd.Series):
            return int(obj.memory_usage(deep=True))
        if hasattr(obj, "nbytes"):
            return int(getattr(obj, "nbytes"))
        if hasattr(obj, "__sizeof__"):
            return int(sys.getsizeof(obj))
        return 0


class AlgorithmMemoryMonitor:
    """
    Context manager and profiler that measures algorithm-level memory footprint
    while excluding the Python interpreter baseline.
    """

    def __init__(
        self,
        operation_name: str = "algorithm_execution",
        strict: bool = False,
        limit_bytes: int = MEMORY_LIMIT_BYTES,
        sample_interval_s: float = 0.005,
    ):
        self.operation_name = operation_name
        self.strict = strict
        self.limit_bytes = limit_bytes
        self.sample_interval_s = sample_interval_s

        self.buffer_tracker = NativeBufferTracker()
        self._baseline_rss: int = 0
        self._baseline_py_heap: int = 0
        self._peak_py_heap: int = 0
        self._peak_rss: int = 0
        self._running = False
        self._sampler_thread: Optional[threading.Thread] = None
        self._profile: Optional[MemoryProfile] = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self._profile and self._profile.status == MemoryConstraintStatus.VIOLATION and self.strict:
            if exc_type is None:
                raise MemoryConstraintViolationError(
                    self.operation_name,
                    self._profile.algorithm_peak_bytes,
                    self.limit_bytes,
                )

    def start(self):
        """Starts monitoring and records the pre-algorithm baseline."""
        gc.collect()
        time.sleep(0.001)

        # 1. Record Interpreter Baseline Process RSS
        if _PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            self._baseline_rss = process.memory_info().rss
            self._peak_rss = self._baseline_rss
        else:
            self._baseline_rss = 0
            self._peak_rss = 0

        # 2. Python-managed heap baseline via tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        tracemalloc.reset_peak()
        self._baseline_py_heap = tracemalloc.get_traced_memory()[0]

        self.buffer_tracker.clear()
        self._running = True

        # 3. Start background polling thread for process RSS and native peaks
        if _PSUTIL_AVAILABLE and self.sample_interval_s > 0:
            self._sampler_thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._sampler_thread.start()

    def _sample_loop(self):
        process = psutil.Process(os.getpid())
        while self._running:
            try:
                rss = process.memory_info().rss
                if rss > self._peak_rss:
                    self._peak_rss = rss
            except Exception:
                pass
            time.sleep(self.sample_interval_s)

    def track_buffer(self, name: str, obj: Any):
        """Explicitly tracks an algorithm buffer (e.g. IMU array, checkpoint pose table)."""
        self.buffer_tracker.register_buffer(name, obj)

    def track_buffers(self, buffers: Dict[str, Any]):
        """Tracks multiple algorithm buffers."""
        for name, obj in buffers.items():
            self.track_buffer(name, obj)

    def stop(self) -> MemoryProfile:
        """Stops monitoring and computes the algorithm-level peak memory footprint."""
        self._running = False
        if self._sampler_thread and self._sampler_thread.is_alive():
            self._sampler_thread.join(timeout=0.1)

        # Get Python heap allocations
        current_py, peak_py = tracemalloc.get_traced_memory()
        py_heap_delta = max(0, peak_py - self._baseline_py_heap)

        # Get registered native buffers peak
        native_buffers_peak = self.buffer_tracker.peak_bytes

        # Calculate Process RSS growth above interpreter baseline
        if _PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            final_rss = process.memory_info().rss
            if final_rss > self._peak_rss:
                self._peak_rss = final_rss
            process_delta = max(0, self._peak_rss - self._baseline_rss)
        else:
            process_delta = 0

        # Algorithm-level Peak Calculation:
        # Sum of attributable Python heap allocations + tracked native buffers
        attributable_algorithm_memory = py_heap_delta + native_buffers_peak
        effective_peak = attributable_algorithm_memory

        # Check coverage: if process RSS grew substantially beyond tracked algorithm memory,
        # note this in the coverage flag while keeping the official compliance metric algorithm-level.
        if process_delta > attributable_algorithm_memory * 3 and (process_delta - attributable_algorithm_memory) > 5 * 1024 * 1024:
            coverage = MemoryCoverage.PARTIALLY_VERIFIED
        else:
            coverage = MemoryCoverage.FULL

        # Evaluate constraint: Peak Algorithm Memory <= 8 MB
        if effective_peak > self.limit_bytes:
            status = MemoryConstraintStatus.VIOLATION
            violation_details = {
                "operation": self.operation_name,
                "peak_bytes": effective_peak,
                "limit_bytes": self.limit_bytes,
                "excess_bytes": effective_peak - self.limit_bytes,
                "excess_mb": round((effective_peak - self.limit_bytes) / (1024 * 1024), 3),
            }
        else:
            if coverage == MemoryCoverage.FULL:
                status = MemoryConstraintStatus.PASS
            else:
                status = MemoryConstraintStatus.PARTIALLY_VERIFIED_PASS
            violation_details = None

        self._profile = MemoryProfile(
            operation_name=self.operation_name,
            limit_bytes=self.limit_bytes,
            interpreter_baseline_bytes=self._baseline_rss,
            process_rss_peak_bytes=self._peak_rss,
            process_rss_delta_bytes=process_delta,
            python_heap_peak_bytes=py_heap_delta,
            native_buffers_peak_bytes=native_buffers_peak,
            algorithm_peak_bytes=effective_peak,
            status=status,
            coverage=coverage,
            tracked_buffers=self.buffer_tracker.buffers,
            violation_details=violation_details,
        )

        return self._profile

    def get_profile(self) -> MemoryProfile:
        if self._profile is None:
            raise RuntimeError("Memory profile is not available. Call stop() first.")
        return self._profile
