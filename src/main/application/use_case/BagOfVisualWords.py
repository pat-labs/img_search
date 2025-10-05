from typing import List

import cv2
import numpy as np
import os
from datetime import datetime

from src.main.application.use_case.ClusterAnalyzer import KNodes
from src.main.application.use_case.ImageUtil import ImageUtil


class BagOfVisualWords:
    KMEANS_ATTEMPTS = 10
    KMEANS_MAX_ITER = 10
    KMEANS_EPSILON = 1.0
    MODEL_FILE_SUFFIX = "_bovw.npz"
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

    def __init__(self, k: KNodes):
        self.k_value = k.value
        self.k_node = k
        self.vocabulary = None
        self.sift = cv2.SIFT_create()

    def create(self, train_data: List[str]):
        all_descriptors = self._extract_sift_features(train_data)
        if all_descriptors:
            self._build_vocabulary(all_descriptors)

    def evaluate(self, test_data_paths: List[str]) -> dict:
        if self.vocabulary is None:
            print("Model has not been trained yet. Cannot evaluate.")
            return None
        result = dict
        for i, (image_path, _) in enumerate(test_data_paths):
            img = ImageUtil.load_grayscale_image(image_path)
            _, descriptors = self.sift.detectAndCompute(img, None)
            if descriptors is not None:
                label = _predict(descriptors)
                result.update({image_path: label})
        return result


    def _extract_sift_features(self, data: list):
        print("Extracting SIFT features...")
        all_descriptors = []
        for i, (image_path, _) in enumerate(data):
            try:
                img = ImageUtil.load_grayscale_image(image_path)
                _, descriptors = self.sift.detectAndCompute(img, None)
                if descriptors is not None:
                    all_descriptors.append(descriptors)
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(data)} images")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        if not all_descriptors:
            print("No descriptors found. Cannot create vocabulary.")
            return None
        return all_descriptors

    def _build_vocabulary(self, all_descriptors):
        print("Stacking descriptors and building vocabulary...")
        stacked_descriptors = np.vstack(all_descriptors).astype(np.float32)

        term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.KMEANS_MAX_ITER, self.KMEANS_EPSILON)
        
        print(f"Performing k-means clustering with k={self.k_value}...")
        _, _, centers = cv2.kmeans(
            stacked_descriptors, self.k_value, None, term_criteria, self.KMEANS_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS
        )

        self.vocabulary = centers
        print(f"Vocabulary of size {self.k_value} created successfully.")

    def get_vocabulary(self):
        return self.vocabulary

    def save_model(self, path: str):
        if self.vocabulary is None:
            print("Model has not been trained yet. Cannot save.")
            return

        os.makedirs(path, exist_ok=True)

        timestamp = datetime.now().strftime(self.TIMESTAMP_FORMAT)
        filename = f"{timestamp}{self.MODEL_FILE_SUFFIX}"
        full_path = os.path.join(path, filename)

        np.savez(full_path, vocabulary=self.vocabulary, k=np.array(self.k_value))

        print(f"Model saved successfully to {full_path}")

    @staticmethod
    def load_model(model_path: str):
        try:
            data = np.load(model_path)
            k_value = int(data['k'])
            vocabulary = data['vocabulary']

            k_node = next((kn for kn in KNodes if kn.value == k_value), None)
            if k_node is None:
                print(f"Error: K value {k_value} from model is not a valid KNodes member.")
                return None

            model = BagOfVisualWords(k=k_node)
            model.vocabulary = vocabulary
            
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None
