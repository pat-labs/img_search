import time
import psutil
import os
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class PerformanceResult:
    memory_usage_mb: float
    execution_time_hours: float

class PerformanceAnalyzer:
    def __init__(self):
        pass


    @staticmethod
    def measure_performance(func: Callable, *args, **kwargs):
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        end_mem = process.memory_info().rss

        execution_time_hours = (end_time - start_time) / 3600
        memory_usage_mb = (end_mem - start_mem) / (1024 * 1024)
        
        return result, PerformanceResult(
            memory_usage_mb=memory_usage_mb,
            execution_time_hours=execution_time_hours
        )
