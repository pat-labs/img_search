
import os
from dataclasses import dataclass
from typing import List
import traceback

from src.main.application.use_case.FileHandler import FileHandler
from src.main.application.use_case.ImageDescriptorAnalyzer import ImageDescriptorAnalyzer, FeatureAlgorithm
from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.application.use_case.PerformanceAnalyzer import PerformanceAnalyzer


@dataclass
class ImageDescriptorStatistic:
    algorithm: str
    variant: str
    match_ratio: float
    avg_distance: float
    memory_usage_mb: float
    execution_time_hours: float


def make_report(image_name: str, image_statistics: List[ImageDescriptorStatistic]) -> str:
    report = f"# Image Descriptor Variance Report for {image_name}\n\n"
    report += "| Algorithm | Variant | Match Ratio | Avg Distance | Memory (MB) | Time (hrs) |\n"
    report += "|---|---|---|---|---|---|\n"
    for stat in image_statistics:
        report += f"| {stat.algorithm} | {stat.variant} | {stat.match_ratio:.2f} | {stat.avg_distance:.2f} | {stat.memory_usage_mb:.4f} | {stat.execution_time_hours:.6f} |\n"
    return report


def test_image_descriptor():
    image_name = "2481823240_eab0d86921.jpg"
    dataset_path = "/home/patrick/Documents/project/img_search/asset/dataset/train/daisy/"
    transform_path = "/home/patrick/Documents/project/img_search/asset/transform_result/"
    report_path = "/home/patrick/Documents/project/img_search/asset/report/"
    full_image_path = os.path.join(dataset_path, image_name)

    ImageUtil.create_image_variances(full_image_path, transform_path)

    image_statistics = []
    image_base_name = os.path.splitext(image_name)[0]
    variant_files = FileHandler.find_files_by_name(transform_path, image_base_name)

    for algorithm in FeatureAlgorithm:
        print(f"--- Testing Algorithm: {algorithm.name} ---")
        try:
            keypoints1, descriptors1 = ImageDescriptorAnalyzer.extract_features(full_image_path, algorithm)
            
            for file_path in variant_files:
                keypoints2, descriptors2 = ImageDescriptorAnalyzer.extract_features(file_path, algorithm)
                
                result, perf_result = PerformanceAnalyzer.measure_performance(
                    ImageDescriptorAnalyzer.calculate_matching, keypoints1, descriptors1, keypoints2, descriptors2
                )
                match_ratio = ImageDescriptorAnalyzer.calculate_match_to_keypoint_ratio(result, keypoints1)
                avg_dist = ImageDescriptorAnalyzer.calculate_average_match_distance(result, keypoints1, keypoints2)
                
                variant_name = os.path.basename(file_path)
                image_statistics.append(ImageDescriptorStatistic(
                    algorithm=algorithm.name,
                    variant=variant_name,
                    match_ratio=match_ratio,
                    avg_distance=avg_dist,
                    memory_usage_mb=perf_result.memory_usage_mb,
                    execution_time_hours=perf_result.execution_time_hours
                ))
        except Exception as e:
            print(f"Error testing {algorithm.name}: {traceback.format_exc()}")

    report = make_report(image_name, image_statistics)
    FileHandler.write_file(report_path, report, extension=".md")
    print(report)

if __name__ == '__main__':
    test_image_descriptor()
