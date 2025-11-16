import os
from typing import Tuple

import cv2
import numpy as np

from src.main.application.use_case.ImageUtil import ImageUtil


class ImageSanitizer:
    @staticmethod
    def noise_reduction(image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    @staticmethod
    def normalize(image):
        norm_img = np.zeros_like(image, dtype=np.uint8)
        return cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

    @staticmethod
    def contrast_enhancement(image):
        if len(image.shape) == 3 and image.shape[2] == 3:  # HSV image
            h, s, v = cv2.split(image)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_v = clahe.apply(v)
            enhanced_hsv = cv2.merge([h, s, enhanced_v])
            return enhanced_hsv
        else:  # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    @staticmethod
    def binarize(image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        return binary_image

    @staticmethod
    def sanitize(img: np.ndarray, image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        try:
            if len(img.shape) == 3:  # If not grayscale, convert it
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            resized_img = cv2.resize(img, image_size)
            denoised_img = ImageSanitizer.noise_reduction(resized_img)
            enhanced_img = ImageSanitizer.contrast_enhancement(denoised_img)
            normalized_img = ImageSanitizer.normalize(enhanced_img)
            return normalized_img
        except Exception as e:
            print(f"Error sanitizing image: {e}")
            raise

    @staticmethod
    def sanitize_dataset(image_paths_and_labels, sanitized_dataset_path, image_size=(224, 224)):
        if not os.path.exists(sanitized_dataset_path):
            os.makedirs(sanitized_dataset_path)

        for image in image_paths_and_labels:
            try:
                img = ImageUtil.load_grayscale_image(image.path)
                diffused_img = cv2.ximgproc.anisotropicDiffusion(img, alpha=0.15, K=30, niters=10)
                diffused_img = (diffused_img * 255).astype(np.uint8)

                resized_img = cv2.resize(diffused_img, image_size)
                denoised_img = ImageSanitizer.noise_reduction(resized_img)
                enhanced_img = ImageSanitizer.contrast_enhancement(denoised_img)
                normalized_img = ImageSanitizer.normalize(enhanced_img) # Normalization should be the last step

                image_name = os.path.basename(image.path)
                sanitized_image_path = os.path.join(sanitized_dataset_path, image_name)
                cv2.imwrite(sanitized_image_path, normalized_img)
            except Exception as e:
                print(f"Error processing image {image.path}: {e}")

        print("Dataset sanitization complete.")