import dataclasses
from typing import List

import cv2 as cv
import os

class ImageUtil:

    @staticmethod
    def load_grayscale_image(image_path: str):
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not open {image_path}")
        return image

    @staticmethod
    def flip_image_horizontally(image):
        return cv.flip(image, 1)

    @staticmethod
    def flip_image_vertically(image):
        return cv.flip(image, 0)

    @staticmethod
    def rotate_image(image, angle):
        (height, width) = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
        return cv.warpAffine(image, rotation_matrix, (width, height))

    @staticmethod
    def change_brightness(image, value):
        hsv = cv.cvtColor(cv.cvtColor(image, cv.COLOR_GRAY2BGR), cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        v = cv.add(v, value)
        final_hsv = cv.merge((h, s, v))
        bright_image = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        return cv.cvtColor(bright_image, cv.COLOR_BGR2GRAY)

    @staticmethod
    def load_image_paths_and_labels(dataset_path: str):
        image_paths_and_labels = []
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Directory not found: {dataset_path}")

        for class_label in os.listdir(dataset_path):
            class_dir = os.path.join(dataset_path, class_label)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    image_paths_and_labels.append({image_path: class_label})
        return image_paths_and_labels

    @staticmethod
    def create_image_variances(image_path: str, result_path: str):
        try:
            original_image = ImageUtil.load_grayscale_image(image_path)
            base_filename, ext = os.path.splitext(os.path.basename(image_path))
            os.makedirs(result_path, exist_ok=True)

            transformations = {
                "rotation_15": (ImageUtil.rotate_image, 15),
                "rotation_45": (ImageUtil.rotate_image, 45),
                "rotation_90": (ImageUtil.rotate_image, 90),
                "brightness_negative_25": (ImageUtil.change_brightness, -25),
                "brightness_25": (ImageUtil.change_brightness, 25),
                "brightness_50": (ImageUtil.change_brightness, 50),
                "flip_horizontal": (ImageUtil.flip_image_horizontally, None),
                "flip_vertical": (ImageUtil.flip_image_vertically, None)
            }

            for suffix, (transform_func, value) in transformations.items():
                if value is not None:
                    transformed_image = transform_func(original_image, value)
                else:
                    transformed_image = transform_func(original_image)
                
                output_filename = f"{base_filename}_{suffix}{ext}"
                output_path = os.path.join(result_path, output_filename)
                cv.imwrite(output_path, transformed_image)
                print(f"Saved {output_path}")

        except Exception as e:
            print(f"An error occurred: {e}")
