import os
import traceback
from dataclasses import dataclass
from typing import List

from src.main.application.use_case.FileHandler import FileHandler
from src.main.application.use_case.ImageSanitizer import ImageSanitizer
from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.application.use_case.PerformanceAnalyzer import PerformanceAnalyzer
from src.main.domain.DescriptorType import DescriptorType


@dataclass
class ImageDescriptorStatistic:
    algorithm: str
    variant: str
    number_keypoints: int
    match_ratio: float
    avg_distance: float
    memory_usage_mb: float
    execution_time_seconds: float


class ImageDescriptorAnalyzer:

    @staticmethod
    def _make_report(image_name: str, image_statistics: List[ImageDescriptorStatistic]) -> str:
        header = "| Algorithm | Variant | Keypoints | Match Ratio | Avg Distance | Memory (MB) | Time (hrs) |\n"
        separator = "|---|---|---|---|---|---|\n"
        rows = [
            f"| {stat.algorithm} | {stat.variant} | {stat.number_keypoints} | {stat.match_ratio:.2f} | {stat.avg_distance:.2f} | {stat.memory_usage_mb:.4f} | {stat.execution_time_seconds:.6f} |"
            for stat in image_statistics]
        return f"# Image Descriptor Variance Report for {image_name}\n\n" + header + separator + "\n".join(rows)

    @staticmethod
    def analyze_image_descriptors(image_path: str, variant_files: List[str]) -> str:
        image_statistics = []
        image_name = os.path.basename(image_path)
        for descriptor_type in DescriptorType:
            print(f"\n--- Testing Algorithm: {descriptor_type.name} ---")
            try:
                features1 = ImageUtil.extract_features(image_path, descriptor_type)
                keypoints1, descriptors1 = features1.keypoints, features1.descriptors
                if keypoints1 is None or len(keypoints1) == 0: continue
                for file_path in variant_files:
                    features2 = ImageUtil.extract_features(file_path, descriptor_type)
                    keypoints2, descriptors2 = features2.keypoints, features2.descriptors
                    if keypoints2 is None or len(keypoints2) == 0: continue
                    matches, perf_result = PerformanceAnalyzer().measure_performance(
                        ImageUtil.calculate_matching, keypoints1, descriptors1, keypoints2, descriptors2)
                    match_ratio = ImageUtil.calculate_match_to_keypoint_ratio(matches, keypoints1)
                    avg_dist = ImageUtil.calculate_average_match_distance(matches, keypoints1, keypoints2)
                    image_statistics.append(
                        ImageDescriptorStatistic(algorithm=descriptor_type.name, variant=os.path.basename(file_path),
                                                 number_keypoints=len(keypoints1),
                                                 match_ratio=match_ratio, avg_distance=avg_dist,
                                                 memory_usage_mb=perf_result.memory_usage_mb,
                                                 execution_time_seconds=perf_result.execution_time_seconds))
            except Exception:
                print(f"Error testing {descriptor_type.name}: {traceback.format_exc()}")
        return ImageDescriptorAnalyzer._make_report(image_name, image_statistics)


if __name__ == '__main__':
    base_path = "/home/patrick/Documents/project/img_search/asset/"
    #image_path = base_path + "dataset/cancer/train/brain_glioma/brain_glioma_0001.jpg"
    image_path = base_path + "dataset/clothes/train/a824deb0-6985-4b11-a987-74d47f5fc33e.jpg"
    #image_path = base_path + "dataset/flowers/train/daisy/2481823240_eab0d86921.jpg"
    report_dir = base_path + "report/"
    transform_dir = base_path + "dataset/variance_clothes/"
    #ImageUtil.create_image_variances(image_path, transform_dir)
    #train_dir = "/home/patrick/Documents/project/img_search/asset/dataset/train"
    images_data = ImageUtil.load_image_data_from_folder(transform_dir)
    #print([img.path for img in images_data])
    report_mk = ImageDescriptorAnalyzer.analyze_image_descriptors(image_path, [img.path for img in images_data])
    FileHandler.write_file(report_mk, report_dir, "", ".md")
