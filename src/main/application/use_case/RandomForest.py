import os
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List

import cv2
import numpy as np

from src.main.application.use_case.ImageUtil import ImageUtil, ImageDataFeature
from src.main.domain.ClassifierType import ClassifierType
from src.main.presentation.ImageDescriptorAnalyzer import DescriptorType


class RandomForest:
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

    def __init__(self, descriptor_type: DescriptorType):
        self.descriptor_type = descriptor_type
        
        self.descriptor = self._create_descriptor()
        self.classifier = cv2.ml.RTrees_create()
        
        self.label_map = None
        
    def is_trained(self) -> bool:
        return self.classifier.isTrained()

    def _create_descriptor(self):
        if self.descriptor_type == DescriptorType.SIFT:
            return cv2.SIFT_create(self.DESIRED_KEYPOINTS)
        elif self.descriptor_type == DescriptorType.ORB:
            return cv2.ORB_create(self.DESIRED_KEYPOINTS)
        elif self.descriptor_type == DescriptorType.KAZE:
            return cv2.KAZE_create()
        elif self.descriptor_type == DescriptorType.AKAZE:
            return cv2.AKAZE_create()
        elif self.descriptor_type == DescriptorType.BRISK:
            return cv2.BRISK_create()
        raise ValueError(f"Unsupported feature algorithm: {self.descriptor_type.name}")

    def _extract_features_for_item(self, item: ImageDataFeature):
        try:
            _, descriptors = self.descriptor.detectAndCompute(ImageUtil.load_grayscale_image(item.path), None)
            if descriptors is not None:
                labels = [item.label] * len(descriptors)
                return descriptors, labels
        except Exception as e:
            print(f"Error processing {item.path}: {e}")
        return None, None

    def train(self, train_data: List[ImageDataFeature], parallel: bool = True):
        print(f"Starting Random Forest training with {self.descriptor_type.name} descriptors...")

        all_descriptors = []
        all_labels = []

        if parallel:
            with ThreadPoolExecutor() as executor:
                future_to_item = {executor.submit(self._extract_features_for_item, item): item for item in train_data}
                for future in as_completed(future_to_item):
                    descriptors, labels = future.result()
                    if descriptors is not None:
                        all_descriptors.extend(descriptors)
                        all_labels.extend(labels)
        else:
            for item in train_data:
                descriptors, labels = self._extract_features_for_item(item)
                if descriptors is not None:
                    all_descriptors.extend(descriptors)
                    all_labels.extend(labels)

        if not all_descriptors:
            print("Training failed: No features could be extracted.")
            return

        print(f"Training Random Forest on {len(all_descriptors)} descriptors...")
        features_np = np.array(all_descriptors, dtype=np.float32)

        label_map = {label: i for i, label in enumerate(np.unique(all_labels))}
        self.label_map = {i: label for label, i in label_map.items()}
        numerical_labels = np.array([label_map[label] for label in all_labels], dtype=np.int32)

        self.classifier.train(features_np, cv2.ml.ROW_SAMPLE, numerical_labels)
        print("Training complete.")

    def predict(self, descriptor: np.ndarray) -> str | None:
        if not self.is_trained():
            print("Prediction failed: The model has not been trained yet.")
            return None

        # Predict the class for each descriptors and find the most common one (voting)
        _, results = self.classifier.predict(descriptor.astype(np.float32))
        predicted_indices = [int(r[0]) for r in results]

        if not predicted_indices:
            return "Unknown"

        most_common_index = Counter(predicted_indices).most_common(1)[0][0]
        return self.label_map.get(most_common_index, "Unknown")
        
    def save_model(self, directory_path: str, base_filename: str = None) -> str | None:
        if not self.is_trained():
            print("Model has not been trained. Cannot save.")
            return None
        
        os.makedirs(directory_path, exist_ok=True)
        
        if base_filename is None:
            timestamp = datetime.now().strftime(self.TIMESTAMP_FORMAT)
            base_filename = f"random_forest_{timestamp}"

        main_model_path = os.path.join(directory_path, f"{base_filename}.pkl")
        
        state = {
            "descriptor_type": self.descriptor_type,
            "label_map": self.label_map,
        }
        
        try:
            with open(main_model_path, "wb") as f:
                pickle.dump(state, f)
            print(f"✅ RandomForest model saved to {main_model_path}")
            return main_model_path
        except Exception as e:
            print(f"❌ Failed to save RandomForest model: {e}")
            return None

    @staticmethod
    def load_model(model_path: str):
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                state = pickle.load(f)
            
            model = RandomForest(
                descriptor_type=state["descriptor_type"],
            )
            model.label_map = state["label_map"]

            model.classifier = cv2.ml.RTrees_create()
            print("📂 Classifier loaded")
            return model
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return None
