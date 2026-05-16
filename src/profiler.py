import time
from contextlib import contextmanager
from typing import Dict, List, Tuple


class PerImageTimer:
    """Track timing for different stages within a single image processing."""

    def __init__(self):
        self.stages: Dict[str, float] = {}

    def record(self, stage_name: str, duration_ms: float):
        """Record the duration of a processing stage."""
        self.stages[stage_name] = duration_ms

    def get_stage(self, stage_name: str) -> float:
        """Retrieve duration of a specific stage."""
        return self.stages.get(stage_name, 0.0)

    def total(self) -> float:
        """Return total time across all stages."""
        return sum(self.stages.values())

    def summary(self) -> str:
        """Return formatted timing breakdown for logging."""
        parts = []
        for stage, time_ms in self.stages.items():
            parts.append(f"{stage}={time_ms:.1f}ms")
        total = self.total()
        return " | ".join(parts) + f" | Total={total:.1f}ms"


class TimingStats:
    """Aggregate statistics for timing across multiple images."""

    def __init__(self):
        self.times: List[float] = []

    def add(self, duration_ms: float):
        """Add a timing measurement."""
        self.times.append(duration_ms)

    def avg(self) -> float:
        """Return average time in milliseconds."""
        return sum(self.times) / len(self.times) if self.times else 0

    def min(self) -> float:
        """Return minimum time in milliseconds."""
        return min(self.times) if self.times else 0

    def max(self) -> float:
        """Return maximum time in milliseconds."""
        return max(self.times) if self.times else 0

    def count(self) -> int:
        """Return number of measurements."""
        return len(self.times)

    def summary(self) -> str:
        """Return formatted statistics summary."""
        if not self.times:
            return "No measurements"
        return f"Avg={self.avg():.1f}ms | Min={self.min():.1f}ms | Max={self.max():.1f}ms | Count={self.count()}"


@contextmanager
def timer():
    """Context manager to measure execution time in milliseconds.
    
    Usage:
        with timer() as t:
            do_something()
        duration_ms = t.elapsed
    """
    class TimerContext:
        def __init__(self):
            self.elapsed = 0.0
            self.start_time = 0.0

    ctx = TimerContext()
    ctx.start_time = time.perf_counter()
    try:
        yield ctx
    finally:
        ctx.elapsed = (time.perf_counter() - ctx.start_time) * 1000
