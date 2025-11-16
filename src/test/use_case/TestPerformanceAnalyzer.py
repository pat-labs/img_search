import time
import unittest

import numpy as np

from src.main.application.use_case.PerformanceAnalyzer import PerformanceAnalyzer, PerformanceResult


def sample_work_function(num_elements):
    time.sleep(0.1)
    data = np.ones(num_elements, dtype=np.float64)
    return data


class TestPerformanceAnalyzer(unittest.TestCase):

    def test_measure_performance(self):
        analyzer = PerformanceAnalyzer()

        result, perf_result = analyzer.measure_performance(sample_work_function, 1_000_000)

        self.assertIsInstance(perf_result, PerformanceResult)
        self.assertGreater(perf_result.execution_time_hours, 0)
        self.assertGreater(perf_result.memory_usage_mb, 0)


if __name__ == '__main__':
    unittest.main()
