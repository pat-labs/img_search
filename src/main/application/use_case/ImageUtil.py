import csv
import multiprocessing
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List

import cv2
import numpy as np

from src.main.application.use_case.AnisotropicSIFT import AnisotropicSIFT
from src.main.application.use_case.SIFTAdHoc import SIFTAdHoc
from src.main.domain.DescriptorType import DescriptorType


@dataclass
class ImageData:
    path: str
    label: str


@dataclass
class ImageDataFeature:
    descriptors: np.ndarray
    keypoints: List[Tuple[float, float]]  # Store keypoint coordinates as tuples
    shape: Tuple[int, int]
    path: str
    label: str


# --- Top-level worker function for multiprocessing ---
def _process_image_worker(args: Tuple[str, str, DescriptorType]) -> Optional[ImageDataFeature]:
    image_path, label, descriptor_type = args
    try:
        descriptor = ImageUtil.create_descriptor(descriptor_type)
        if descriptor is None: return None
        img = ImageUtil.load_grayscale_image(image_path)
        h, w = img.shape
        keys, desc = descriptor.detectAndCompute(img, None)
        if desc is not None and len(desc) > 0:
            # Convert KeyPoint objects to a serializable format (tuples of coordinates)
            keypoint_coords = [kp.pt for kp in keys]
            return ImageDataFeature(desc.astype(np.float32), keypoint_coords, (h, w), image_path, label)
        else:
            print(f"⚠️ No descriptors for {image_path}")
    except Exception as e:
        print(f"Error processing {image_path} in worker: {e}")
        return None

DESIRED_KEYPOINTS = 5_000

class ImageUtil:

    @staticmethod
    def load_grayscale_image(image_path: str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not open {image_path}")
        return image

    @staticmethod
    def _flip_image_horizontally(image):
        return cv2.flip(image, 1)

    @staticmethod
    def _flip_image_vertically(image):
        return cv2.flip(image, 0)

    @staticmethod
    def _rotate_image(image, angle):
        (height, width) = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))

    @staticmethod
    def _change_brightness(image, value):
        hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        final_hsv = cv2.merge((h, s, v))
        bright_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _scale_image(image, factor):
        height, width = image.shape[:2]
        new_height, new_width = int(height * factor), int(width * factor)
        interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    @staticmethod
    def _blur_image(image, kernel_size: Tuple[int, int]):
        return cv2.GaussianBlur(image, kernel_size, 0)

    @staticmethod
    def load_image_data_from_folder(dataset_path: str):
        image_paths_and_labels = []
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Directory not found: {dataset_path}")

        dir_items = os.listdir(dataset_path)
        contains_subdirs = any(os.path.isdir(os.path.join(dataset_path, item)) for item in dir_items)

        if contains_subdirs:
            for class_label in dir_items:
                class_dir = os.path.join(dataset_path, class_label)
                if os.path.isdir(class_dir):
                    for image_name in os.listdir(class_dir):
                        image_path = os.path.join(class_dir, image_name)
                        image_paths_and_labels.append(ImageData(path=image_path, label=class_label))
        else:
            for image_name in dir_items:
                image_path = os.path.join(dataset_path, image_name)
                if os.path.isfile(image_path):
                    image_paths_and_labels.append(ImageData(path=image_path, label="Unknown"))

        return image_paths_and_labels

    @staticmethod
    def load_image_data_from_csv(file_path: str):
        data_list = []
        with open(file_path, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                data_list.append(ImageData(path=row[0], label=row[1]))
        return data_list

    @staticmethod
    def create_image_variances(image_path: str, result_path: str):
        try:
            original_image = ImageUtil.load_grayscale_image(image_path)
            base_filename, ext = os.path.splitext(os.path.basename(image_path))

            transformations = {
                "rotation": {
                    "rotation_15": (ImageUtil._rotate_image, 15),
                    "rotation_45": (ImageUtil._rotate_image, 45),
                    "rotation_90": (ImageUtil._rotate_image, 90)
                },
                "brightness": {
                    "brightness_negative_25": (ImageUtil._change_brightness, -25),
                    "brightness_25": (ImageUtil._change_brightness, 25),
                    "brightness_50": (ImageUtil._change_brightness, 50)
                },
                "flip": {
                    "flip_horizontal": (ImageUtil._flip_image_horizontally, None),
                    "flip_vertical": (ImageUtil._flip_image_vertically, None)
                },
                "scale": {
                    "scale_up_1_5x": (ImageUtil._scale_image, 1.5),
                    "scale_down_0_75x": (ImageUtil._scale_image, 0.75)
                },
                "blur": {
                    "blur_light": (ImageUtil._blur_image, (3, 3)),
                    "blur_medium": (ImageUtil._blur_image, (7, 7))
                },
            }

            os.makedirs(result_path, exist_ok=True)
            for folder, funcs in transformations.items():
                folder_path = os.path.join(result_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                for suffix, (transform_func, value) in funcs.items():
                    if value is not None:
                        transformed_image = transform_func(original_image, value)
                    else:
                        transformed_image = transform_func(original_image)

                    output_filename = f"{base_filename}_{suffix}{ext}"
                    output_path = os.path.join(folder_path, output_filename)
                    cv2.imwrite(output_path, transformed_image)
                    print(f"Saved {output_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

    @staticmethod
    def calculate_matching(keypoints1, descriptors1, keypoints2, descriptors2):
        if not all(x is not None and len(x) > 0 for x in [keypoints1, descriptors1, keypoints2, descriptors2]):
            return []
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)
        return sorted(matches, key=lambda x: x.distance)

    @staticmethod
    def calculate_match_to_keypoint_ratio(matches, keypoints1):
        return len(matches) / len(keypoints1) if keypoints1 else 0.0

    @staticmethod
    def calculate_average_match_distance(matches, keypoints1, keypoints2):
        """
        Calculates the average distance between matched keypoints.
        Assumes keypoints are provided as lists of (x, y) coordinate tuples.
        """
        if not matches:
            return 0.0

        distances = []
        for m in matches:
            if m.queryIdx < len(keypoints1) and m.trainIdx < len(keypoints2):
                # Handle both cv2.KeyPoint objects and coordinate tuples
                kp1 = keypoints1[m.queryIdx]
                kp2 = keypoints2[m.trainIdx]
                
                pt1 = kp1.pt if isinstance(kp1, cv2.KeyPoint) else kp1
                pt2 = kp2.pt if isinstance(kp2, cv2.KeyPoint) else kp2

                distances.append(np.linalg.norm(np.array(pt1) - np.array(pt2)))

        return float(np.mean(distances)) if distances else 0.0

    @staticmethod
    def extract_features(image_path: str, algorithm: DescriptorType, label: str = "Unknown"):
        descriptor = ImageUtil.create_descriptor(algorithm)
        if descriptor is None: return [], None
        img = ImageUtil.load_grayscale_image(image_path)
        h, w = img.shape
        keys, desc = descriptor.detectAndCompute(img, None)
        if desc is not None and len(desc) > 0:
            return ImageDataFeature(desc.astype(np.float32), keys, (h, w), image_path, label)
        else:
            print(f"⚠️ No descriptors for {image_path}")
            return [], None

    @staticmethod
    def extract_descriptors_parallel(path_labels: List[ImageData], algorithm: DescriptorType):
        worker_args = [(item.path, item.label, algorithm) for item in path_labels]
        with multiprocessing.Pool() as pool:
            results = pool.map(_process_image_worker, worker_args)
        return [r for r in results if r is not None]

    @staticmethod
    def extract_descriptors_serial(path_labels: List[ImageData], algorithm: DescriptorType):
        results = []
        for item in path_labels:
            results.append(ImageUtil.extract_features(item.path, algorithm))
        return results

    @staticmethod
    def create_descriptor(descriptor_type):
        if descriptor_type == DescriptorType.SIFT:
            return cv2.SIFT_create(DESIRED_KEYPOINTS)
        elif descriptor_type == DescriptorType.ORB:
            return cv2.ORB_create(DESIRED_KEYPOINTS)
        elif descriptor_type == DescriptorType.KAZE:
            return cv2.KAZE_create()
        elif descriptor_type == DescriptorType.AKAZE:
            return cv2.AKAZE_create()
        elif descriptor_type == DescriptorType.BRISK:
            return cv2.BRISK_create()
        elif descriptor_type == DescriptorType.ANSIOTROPIC_SIFT:
            return AnisotropicSIFT()
        elif descriptor_type == DescriptorType.SIFT_ADHOC:
            return SIFTAdHoc()
        raise ValueError(f"Unsupported feature algorithm: {descriptor_type.name}")