import os
from enum import Enum
import time
import psutil
from dataclasses import dataclass

import cv2
import numpy as np

from src.main.application.use_case.FisherMatrix import FisherMatrix
from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.application.use_case.BagOfVisualWords import BagOfVisualWords
from src.main.application.use_case.Kmeans import KMeans
from src.main.application.use_case.PerformanceAnalyzer import PerformanceAnalyzer


class KNodes(Enum):
    K16 = 16
    K32 = 32
    K64 = 64
    K128 = 128
    K256 = 256

class ClusterAlgorithm(Enum):
    BAG_OF_VISUAL_WORDS = 1
    FISHER_MATRIX = 2
    CUSTOM_KMEANS = 3

@dataclass
class ClusterStatistic:
    algorithm: ClusterAlgorithm
    accuracy: float
    memory_usage_mb: float
    execution_time_hours: float


class ClusterAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.train_dir = os.path.join(self.dataset_path, "train")
        self.test_dir = os.path.join(self.dataset_path, "test")
        self.model_dir = os.path.join(self.dataset_path, "model")
        self.train_dataset = ImageUtil.load_image_paths_and_labels(self.train_dir)
        self.test_dataset = ImageUtil.load_image_paths_and_labels(self.test_dir)
        self.performance_results = []

    def train_size(self):
        size = len(self.train_dataset)
        print(f"Number of training images: {size}")
        return size

    def test_size(self):
        size = len(self.test_dataset)
        print(f"Number of testing images: {size}")
        return size

    def analyze_cluster_algorithm(self, k: KNodes):
        bag_of_visual_words = BagOfVisualWords(k)
        _, train_perf_result = PerformanceAnalyzer.measure_performance(
        bag_of_visual_words.create, self.train_dataset)
        test_result_paths_and_labess, test_perf_result = PerformanceAnalyzer.measure_performance(
        bag_of_visual_words.evaluate, self.test_dataset)
        test_path_and_labels = ImageUtil.to_dict(self.test_dataset)
        accuracy =

        self.performance_results.append(ClusterStatistic(
            algorithm=ClusterAlgorithm.BAG_OF_VISUAL_WORDS,
            accuracy=0.0,  # Placeholder
            memory_usage_mb=perf_train_result.memory_usage_mb,
            execution_time_hours=perf_train_result.execution_time_hours
        ))

        fisher_matrix = FisherMatrix(k)
        _, perf_result = PerformanceAnalyzer.measure_performance(
        fisher_matrix.create, self.train_dataset)
        self.performance_results.append(ClusterStatistic(
            algorithm=ClusterAlgorithm.FISHER_MATRIX,
            accuracy=0.0,  # Placeholder
            memory_usage_mb=perf_result.memory_usage_mb,
            execution_time_hours=perf_result.execution_time_hours
        ))

        # --- Custom KMeans ---
        print("--- Running Custom KMeans ---")
        sift = cv2.SIFT_create()
        all_descriptors = []
        print("Extracting SIFT features for custom KMeans...")
        for i, (image_path, _) in enumerate(self.train_dataset):
            try:
                img = ImageUtil.load_grayscale_image(image_path)
                _, descriptors = sift.detectAndCompute(img, None)
                if descriptors is not None:
                    all_descriptors.append(descriptors)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        if all_descriptors:
            stacked_descriptors = np.vstack(all_descriptors).astype(np.float32)
            
            start_mem = process.memory_info().rss
            start_time = time.time()

            print(f"Performing k-means clustering with k={k.value} using custom KMeans...")
            kmeans = KMeans(n_clusters=k.value)
            kmeans.fit(stacked_descriptors)
            
            end_time = time.time()
            end_mem = process.memory_info().rss
            self.performance_results.append(AlgorithmPerformance(
                name="CustomKMeans",
                precision=0.0,  # Placeholder
                time_cost=end_time - start_time,
                memory_cost=(end_mem - start_mem) / (1024 * 1024)
            ))
            print("Custom KMeans finished.")
        else:
            print("No descriptors found for custom KMeans.")

    def make_report(self) -> str:
        if not self.performance_results:
            return "No performance results to report. Run `analyze_cluster_algorithm` first."

        report = "# Clustering Algorithm Performance Report\n\n"
        report += "| Algorithm | Precision | Time Cost (s) | Memory Cost (MB) |\n"
        report += "|---|---|---|---|\n"

        for result in self.performance_results:
            report += f"| {result.name} | {result.precision:.2f} | {result.time_cost:.2f} | {result.memory_cost:.2f} |\n"
        
        return report
