import unittest
import os
import tempfile
import shutil
import numpy as np
import cv2
import multiprocessing

from src.main.application.use_case.ImageDescriptorAnalyzer import ImageDescriptorAnalyzer, DescriptorType
from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.application.use_case.FileHandler import FileHandler
from src.main.application.use_case.PerformanceAnalyzer import PerformanceAnalyzer

class TestImageDescriptorAnalyzer(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.source_image_path = os.path.join(self.test_dir, "source.png")
        
        # Use a larger image and more images to make the performance difference measurable
        dummy_image = np.random.randint(0, 256, (250, 250), dtype=np.uint8)
        cv2.imwrite(self.source_image_path, dummy_image)
        
        ImageUtil.create_image_variances(self.source_image_path, self.test_dir)

        self.perf_test_paths = []
        for i in range(50):
            path = os.path.join(self.test_dir, f"perf_img_{i}.png")
            cv2.imwrite(path, dummy_image)
            self.perf_test_paths.append(path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_analyze_image_descriptors_report(self):
        image_base_name = os.path.splitext(os.path.basename(self.source_image_path))[0]
        variant_files = FileHandler.find_files_by_name(self.test_dir, image_base_name)
        variant_files = [p for p in variant_files if p != self.source_image_path]

        self.assertGreater(len(variant_files), 0)

        report = ImageDescriptorAnalyzer.analyze_image_descriptors(
            image_path=self.source_image_path, 
            variant_files=variant_files
        )

        self.assertIsInstance(report, str)
        self.assertNotEqual(report, "")
        self.assertIn("# Image Descriptor Variance Report", report)
        self.assertTrue(any(algo.name in report for algo in DescriptorType))

    def test_serial_vs_parallel_performance(self):
        perf_analyzer = PerformanceAnalyzer()

        for algorithm_to_test in DescriptorType:
            serial_descriptors, serial_result = perf_analyzer.measure_performance(
                ImageDescriptorAnalyzer.extract_features_serial, self.perf_test_paths, algorithm_to_test
            )

            parallel_descriptors, parallel_result = perf_analyzer.measure_performance(
                ImageDescriptorAnalyzer.extract_features_parallel, self.perf_test_paths, algorithm_to_test
            )

            print("\n--- Serial vs. Parallel Performance Comparison ---")
            print(f"Serial Time:   {serial_result.execution_time_hours * 3600:.4f}s, Memory: {serial_result.memory_usage_mb:.2f}MB")
            print(f"Parallel Time: {parallel_result.execution_time_hours * 3600:.4f}s, Memory: {parallel_result.memory_usage_mb:.2f}MB")

            self.assertIsNotNone(serial_descriptors)
            self.assertIsNotNone(parallel_descriptors)
            self.assertGreater(len(serial_descriptors), 0)
            self.assertGreater(len(parallel_descriptors), 0)

            if multiprocessing.cpu_count() > 1:
                self.assertLess(parallel_result.execution_time_hours, serial_result.execution_time_hours,
                              "Parallel execution should be faster than serial on a multi-core machine.")

if __name__ == '__main__':
    unittest.main()
