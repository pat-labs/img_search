import os
import traceback
import multiprocessing
from dataclasses import dataclass
from typing import List, Tuple

import cv2 as cv
import numpy as np

from src.main.application.use_case.DescriptorType import DescriptorType
from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.application.use_case.PerformanceAnalyzer import PerformanceAnalyzer


# --- Top-level worker function for multiprocessing ---
def _process_image_worker(args: Tuple[str, DescriptorType]) -> np.ndarray | None:
    image_path, algorithm = args
    try:
        descriptor = algorithm.create_descriptor()
        if descriptor is None: return None
        img = ImageUtil.load_grayscale_image(image_path)
        _, desc = descriptor.detectAndCompute(img, None)
        return desc
    except Exception as e:
        print(f"Error processing {image_path} in worker: {e}")
        return None


@dataclass
class ImageDescriptorStatistic:
    algorithm: str
    variant: str
    match_ratio: float
    avg_distance: float
    memory_usage_mb: float
    execution_time_hours: float


class ImageDescriptorAnalyzer:

    @staticmethod
    def calculate_matching(keypoints1, descriptors1, keypoints2, descriptors2):
        if not all(x is not None and len(x) > 0 for x in [keypoints1, descriptors1, keypoints2, descriptors2]):
            return []
        matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)
        return sorted(matches, key=lambda x: x.distance)

    @staticmethod
    def calculate_match_to_keypoint_ratio(matches, keypoints1):
        return len(matches) / len(keypoints1) if keypoints1 else 0.0

    @staticmethod
    def calculate_average_match_distance(matches, keypoints1, keypoints2):
        if not matches: return 0.0
        distances = [np.linalg.norm(np.array(keypoints1[m.queryIdx].pt) - np.array(keypoints2[m.trainIdx].pt)) for m in matches]
        return float(np.mean(distances)) if distances else 0.0

    @staticmethod
    def extract_features(image_path: str, algorithm: DescriptorType, desired_keypoints: int = 5000) -> Tuple[list, np.ndarray]:
        image = ImageUtil.load_grayscale_image(image_path)
        detector, descriptor = algorithm.create_descriptor(desired_keypoints)
        if detector is None: return [], None
        if descriptor is not None:
            keypoints = detector.detect(image, None)
            return descriptor.compute(image, keypoints)
        return detector.detectAndCompute(image, None)

    @staticmethod
    def _make_report(image_name: str, image_statistics: List[ImageDescriptorStatistic]) -> str:
        header = "| Algorithm | Variant | Match Ratio | Avg Distance | Memory (MB) | Time (hrs) |\n"
        separator = "|---|---|---|---|---|---|\n"
        rows = [f"| {stat.algorithm} | {stat.variant} | {stat.match_ratio:.2f} | {stat.avg_distance:.2f} | {stat.memory_usage_mb:.4f} | {stat.execution_time_hours:.6f} |" for stat in image_statistics]
        return f"# Image Descriptor Variance Report for {image_name}\n\n" + header + separator + "\n".join(rows)

    @staticmethod
    def analyze_image_descriptors(image_path: str, variant_files: List[str]) -> str:
        image_statistics = []
        image_name = os.path.basename(image_path)
        for algorithm in DescriptorType:
            print(f"\n--- Testing Algorithm: {algorithm.name} ---")
            try:
                keypoints1, descriptors1 = ImageDescriptorAnalyzer.extract_features(image_path, algorithm)
                if keypoints1 is None or len(keypoints1) == 0: continue
                for file_path in variant_files:
                    keypoints2, descriptors2 = ImageDescriptorAnalyzer.extract_features(file_path, algorithm)
                    matches, perf_result = PerformanceAnalyzer().measure_performance(ImageDescriptorAnalyzer.calculate_matching, keypoints1, descriptors1, keypoints2, descriptors2)
                    match_ratio = ImageDescriptorAnalyzer.calculate_match_to_keypoint_ratio(matches, keypoints1)
                    avg_dist = ImageDescriptorAnalyzer.calculate_average_match_distance(matches, keypoints1, keypoints2)
                    image_statistics.append(ImageDescriptorStatistic(algorithm=algorithm.name, variant=os.path.basename(file_path), match_ratio=match_ratio, avg_distance=avg_dist, memory_usage_mb=perf_result.memory_usage_mb, execution_time_hours=perf_result.execution_time_hours))
            except Exception:
                print(f"Error testing {algorithm.name}: {traceback.format_exc()}")
        return ImageDescriptorAnalyzer._make_report(image_name, image_statistics)

    @staticmethod
    def extract_features_parallel(image_paths: List[str], algorithm: DescriptorType) -> List[np.ndarray]:
        worker_args = [(path, algorithm) for path in image_paths]
        with multiprocessing.Pool() as pool:
            results = pool.map(_process_image_worker, worker_args)
        return [d for descs in results if descs is not None for d in descs]

    @staticmethod
    def extract_features_serial(image_paths: List[str], algorithm: DescriptorType):
        print(f"Extracting {algorithm.name} features...")
        descriptors = []
        descriptor = algorithm.create_descriptor()
        if descriptor is None: return None
        for path in image_paths:
            try:
                img = ImageUtil.load_grayscale_image(path)
                _, desc = descriptor.detectAndCompute(img, None)
                if desc is not None:
                    descriptors.append(desc)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        return descriptors if descriptors else []
