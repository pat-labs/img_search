import os
import cv2
import numpy as np

from src.main.application.use_case.ImageUtil import ImageUtil


class PrepareDataset:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.train_dir = os.path.join(self.dataset_path, "train")
        self.test_dir = os.path.join(self.dataset_path, "test")
        self.sanitized_dataset_path = os.path.join(self.dataset_path, "sanitized_dataset")

    def train_size(self):
        train_data = self.load_dataset(self.train_dir)
        size = len(train_data)
        print(f"Number of training images: {size}")
        return size

    def test_size(self):
        test_data = self.load_dataset(self.test_dir)
        size = len(test_data)
        print(f"Number of testing images: {size}")
        return size

    def sanitize_dataset(self, image_size=(224, 224)):
        if not os.path.exists(self.sanitized_dataset_path):
            os.makedirs(self.sanitized_dataset_path)

        for dataset_type in ["train", "test"]:
            original_dataset_path = os.path.join(self.dataset_path, dataset_type)
            sanitized_dataset_type_path = os.path.join(self.sanitized_dataset_path, dataset_type)

            if not os.path.exists(sanitized_dataset_type_path):
                os.makedirs(sanitized_dataset_type_path)

            image_paths_and_labels = self.load_dataset(original_dataset_path)

            for image_path, class_label in image_paths_and_labels:
                try:
                    img = ImageUtil._load_grayscale_image(image_path)

                    resized_img = cv2.resize(img, image_size)
                    denoised_img = self.noise_reduction(resized_img)
                    hsv_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2HSV)
                    enhanced_img = self.contrast_enhancement(hsv_img)
                    normalized_img = self.normalize(enhanced_img)

                    target_class_dir = os.path.join(sanitized_dataset_type_path, class_label)
                    if not os.path.exists(target_class_dir):
                        os.makedirs(target_class_dir)

                    image_name = os.path.basename(image_path)
                    sanitized_image_path = os.path.join(target_class_dir, image_name)
                    cv2.imwrite(sanitized_image_path, normalized_img)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

        print("Dataset sanitization complete.")

    @staticmethod
    def load_dataset(dataset_path: str):
        image_paths_and_labels = []
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Directory not found: {dataset_path}")

        for class_label in os.listdir(dataset_path):
            class_dir = os.path.join(dataset_path, class_label)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    image_paths_and_labels.append((image_path, class_label))
        return image_paths_and_labels
