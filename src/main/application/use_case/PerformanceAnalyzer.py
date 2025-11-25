import os
import time
from dataclasses import dataclass
from typing import Callable
import tracemalloc

import psutil


@dataclass
class PerformanceResult:
    memory_usage_mb: float
    execution_time_seconds: float


class PerformanceAnalyzer:

    @staticmethod
    def measure_performance(func: Callable, *args, **kwargs):
        # Use tracemalloc for a more accurate measure of peak memory allocated by the function
        tracemalloc.start()

        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        # Get the current and peak memory usage since tracemalloc.start()
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_time_seconds = end_time - start_time
        # The peak memory usage is a more reliable metric than start/end difference
        memory_usage_mb = peak_mem_bytes / (1024 * 1024)

        return result, PerformanceResult(
            # Ensure memory usage is not reported as negative
            memory_usage_mb=max(0.0, memory_usage_mb),
            execution_time_seconds=execution_time_seconds
        )
