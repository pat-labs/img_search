import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import cv2
import numpy as np

from src.main.application.use_case.ImageUtil import ImageDataFeature
from src.main.domain.ClassifierType import ClassifierType
from src.main.domain.DescriptorType import DescriptorType
from src.main.domain.Knodes import KNodes


class BagOfVisualWords:
    KMEANS_ATTEMPTS = 10
    KMEANS_MAX_ITER = 10
    KMEANS_EPSILON = 1.0
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    DESIRED_KEYPOINTS = 5_000

    def __init__(self, k: KNodes,
                 classifier_type: ClassifierType,
                 descriptor_type: DescriptorType):
        self.k = k
        self.classifier_type = classifier_type
        self.descriptor_type = descriptor_type

        self.classifier = self._create_classifier()
        self.descriptor = self._create_descriptor()
        self.matcher = self._create_matcher()

        self.vocabulary = None
        self.label_map = None

    def is_trained(self) -> bool:
        return self.vocabulary is not None and self.classifier.isTrained()

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

    def _create_matcher(self):
        if self.descriptor_type in [DescriptorType.SIFT, DescriptorType.KAZE, DescriptorType.AKAZE, DescriptorType.BRISK]:
            return cv2.BFMatcher(cv2.NORM_L2)
        elif self.descriptor_type == DescriptorType.ORB:
            return cv2.BFMatcher(cv2.NORM_HAMMING)
        raise ValueError(f"Unsupported matcher for algorithm: {self.descriptor_type.name}")

    def _create_classifier(self):
        if self.classifier_type == ClassifierType.SVM:
            return cv2.ml.SVM_create()
        elif self.classifier_type == ClassifierType.LOGISTIC_REGRESSION:
            return cv2.ml.LogisticRegression_create()
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type.name}")

    def train(self, train_data: list[ImageDataFeature], parallel: bool = True):
        valid_desc = [item.descriptor for item in train_data if
                      item.descriptor is not None and len(item.descriptor) > 0]
        if not valid_desc:
            raise ValueError("No valid descriptors provided for GMM training.")

        all_descriptors = np.vstack(valid_desc).astype(np.float32)
        self._build_vocabulary(all_descriptors)

        if parallel:
            histograms, labels = self._compute_histogram_parallel(train_data)
        else:
            histograms, labels = self._compute_histogram_serial(train_data)

        unique_labels = sorted(list(set(labels)))
        self.label_map = {i: label for i, label in enumerate(unique_labels)}
        self.inverse_label_map = {label: i for i, label in enumerate(unique_labels)}
        numerical_labels = np.array([self.inverse_label_map[lbl] for lbl in labels], dtype=np.int32)

        self.classifier.train(np.array(histograms, dtype=np.float32), cv2.ml.ROW_SAMPLE,
                              numerical_labels)
        print("Training complete.")

    def _compute_histogram_serial(self, train_data: list[ImageDataFeature]):
        histograms, labels = [], []
        for item in train_data:
            hist = self._generate_histogram(item.descriptor)
            if hist is not None and hist.size > 0:
                histograms.append(hist)
                labels.append(item.label)
        return histograms, labels

    def _compute_histogram_parallel(self, train_data: list[ImageDataFeature]):
        histograms, labels = [], []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._generate_histogram, item.descriptor): item.label for item in train_data}
            for future in as_completed(futures):
                hist = future.result()
                if hist is not None and hist.size > 0:
                    histograms.append(hist)
                    labels.append(futures[future])
        return histograms, labels

    def predict(self, descriptor: np.ndarray) -> str | None:
        if not self.is_trained():
            print("Prediction failed: The model has not been trained yet.")
            return None

        histogram = self._generate_histogram(descriptor)
        if histogram is None or histogram.size == 0:
            return None

        _, result = self.classifier.predict(np.array([histogram], dtype=np.float32))
        predicted_label_index = int(result[0][0])
        return self.label_map.get(predicted_label_index, "Unknown")

    def _generate_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        histogram = np.zeros(self.k.value, dtype=np.float32)

        # Ensure consistent types for matching
        if self.descriptor_type == DescriptorType.ORB:
            # ORB uses binary descriptors (uint8)
            descriptors = descriptors.astype(np.uint8)
            vocabulary = np.uint8(np.clip(self.vocabulary, 0, 255))
        else:
            # SIFT, KAZE, AKAZE, BRISK use float32
            descriptors = descriptors.astype(np.float32)
            vocabulary = self.vocabulary.astype(np.float32)

        matches = self.matcher.match(descriptors, vocabulary)
        for match in matches:
            histogram[match.trainIdx] += 1
        return histogram

    def _build_vocabulary(self, all_descriptors):
        print("Building vocabulary with k-means...")
        term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.KMEANS_MAX_ITER, self.KMEANS_EPSILON)

        _, _, centers = cv2.kmeans(
            all_descriptors, self.k.value, None, term_criteria, self.KMEANS_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS
        )
        self.vocabulary = centers
        print(f"Vocabulary of size {self.k.value} created.")

    def save_model(self, directory_path: str, file_name: str = None) -> str | None:
        if not self.is_trained():
            print("Model has not been trained. Cannot save.")
            return None

        os.makedirs(directory_path, exist_ok=True)
        if file_name is None:
            timestamp = datetime.now().strftime(self.TIMESTAMP_FORMAT)
            file_name = f"bovw_{timestamp}"

        classifier_path = os.path.join(directory_path, f"{file_name}_classifier.xml")
        full_path = os.path.join(directory_path, f"{file_name}.pkl")
        try:
            self.classifier.save(classifier_path)
            print(f"💾 Classifier saved to {classifier_path}")
        except Exception as e:
            print(f"❌ Failed to save classifier: {e}")
            return None

        state = {
            "k": self.k,
            "classifier_type": self.classifier_type,
            "descriptor_type": self.descriptor_type,
            "vocabulary": self.vocabulary,
            "label_map": self.label_map,
            "inverse_label_map": self.inverse_label_map,
        }

        try:
            with open(full_path, "wb") as f:
                pickle.dump(state, f)
            print(f"💾 Model (vocabulary + metadata) saved to {full_path}")
        except Exception as e:
            print(f"❌ Failed to save pickle model: {e}")
            return None

        return full_path

    @staticmethod
    def load_model(model_path: str):
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None

        # ✅ Extract components correctly
        directory_path = os.path.dirname(model_path)
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        classifier_path = os.path.join(directory_path, f"{base_name}_classifier.xml")

        try:
            with open(model_path, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            print(f"❌ Failed to load pickle file: {e}")
            return None

        try:
            # ✅ Reconstruct model
            model = BagOfVisualWords(
                state["k"],
                state["classifier_type"],
                state["descriptor_type"]
            )

            model.vocabulary = state["vocabulary"]
            model.label_map = state["label_map"]
            model.inverse_label_map = state["inverse_label_map"]

            # ✅ Reload classifier
            if state["classifier_type"].name == "SVM":
                model.classifier = cv2.ml.SVM_load(classifier_path)
            elif state["classifier_type"].name == "LOGISTIC_REGRESSION":
                model.classifier = cv2.ml.LogisticRegression_load(classifier_path)
            else:
                raise ValueError(f"Unsupported classifier type: {state['classifier_type'].name}")

            print(f"📂 Classifier loaded from {classifier_path}")
            return model

        except Exception as e:
            print(f"❌ Failed to rebuild BagOfVisualWords: {e}")
            return None
