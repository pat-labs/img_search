import cv2
import numpy as np
import os
from datetime import datetime
from enum import Enum
from typing import Optional

from src.main.application.use_case.ClassifierType import ClassifierType
from src.main.application.use_case.DescriptorType import DescriptorType
from src.main.application.use_case.ImageUtil import ImageUtil, PathLabel

class KNodes(Enum):
    K16 = 16
    K32 = 32
    K64 = 64
    K128 = 128
    K256 = 256

class BagOfVisualWords:
    KMEANS_ATTEMPTS = 10
    KMEANS_MAX_ITER = 10
    KMEANS_EPSILON = 1.0
    MODEL_FILE_SUFFIX = "_bovw.npz"
    CLASSIFIER_FILE_SUFFIX = "_bovw_classifier.xml"
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

    def __init__(self, k: KNodes, 
                 classifier_type: ClassifierType,
                 descriptor_type: DescriptorType):
        self.k_value = k.value
        self.k_node = k
        self.classifier_type = classifier_type
        self.descriptor_type = descriptor_type
        self.vocabulary = None
        self.classifier = classifier_type.create_classifier()
        self.descriptor = descriptor_type.create_descriptor()
        self.detector = self.descriptor
        self.descriptor_dtype = descriptor_type.get_descriptor_dtype()
        self.matcher = descriptor_type.create_matcher()
        self.label_map = None

    def is_trained(self) -> bool:
        return self.vocabulary is not None and self.classifier.isTrained()

    def train(self, train_data: list[PathLabel], precomputed_descriptors: list[np.ndarray]):
        print(f"Starting training with {self.descriptor_type.name} and {self.classifier_type.name}...")

        if precomputed_descriptors is None or len(precomputed_descriptors) == 0:
            print("Training failed: No features available for vocabulary.")
            return
        
        # Stack all descriptors into a single numpy array for k-means
        all_descriptors_stacked = np.vstack(precomputed_descriptors)
        all_descriptors_stacked = all_descriptors_stacked.astype(self.descriptor_dtype)
        self._build_vocabulary(all_descriptors_stacked)

        print("Creating histograms for training images...")
        
        # Process histograms and labels together to ensure they stay in sync
        train_histograms = []
        train_labels = []
        for item, descs in zip(train_data, precomputed_descriptors):
            if descs is not None and len(descs) > 0:
                train_histograms.append(self._generate_histogram(descs.astype(self.descriptor_dtype)))
                train_labels.append(item.label)

        print(f"Training the {self.classifier_type.name} classifier...")
        label_map = {label: i for i, label in enumerate(np.unique(train_labels))}
        self.label_map = {i: label for label, i in label_map.items()}
        numerical_labels = np.array([label_map[label] for label in train_labels], dtype=np.int32)

        self.classifier.train(np.array(train_histograms, dtype=self.descriptor_dtype), cv2.ml.ROW_SAMPLE, numerical_labels)
        print("Training complete.")

    def predict(self, image_path: str) -> str | None:
        if not self.is_trained():
            print("Prediction failed: The model has not been trained yet.")
            return None
        
        img = ImageUtil.load_grayscale_image(image_path)
        _, descs = self.detector.detectAndCompute(img, None)

        if descs is None:
            return "Unknown" # Or handle as you see fit

        histogram = self._generate_histogram(descs.astype(self.descriptor_dtype))
        _, result = self.classifier.predict(np.array([histogram], dtype=self.descriptor_dtype))
        predicted_label_index = int(result[0][0])
        
        return self.label_map.get(predicted_label_index, "Unknown")

    def _generate_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        histogram = np.zeros(self.k_value, dtype=self.descriptor_dtype)

        if descriptors is not None:
            matches = self.matcher.match(descriptors, self.vocabulary)
            for match in matches:
                histogram[match.trainIdx] += 1
        return histogram

    def _build_vocabulary(self, all_descriptors):
        print("Building vocabulary with k-means...")
        term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.KMEANS_MAX_ITER, self.KMEANS_EPSILON)
        
        _, _, centers = cv2.kmeans(
            all_descriptors, self.k_value, None, term_criteria, self.KMEANS_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS
        )
        self.vocabulary = centers
        print(f"Vocabulary of size {self.k_value} created.")

    def save_model(self, path: str):
        if not self.is_trained():
            print("Model has not been trained yet. Cannot save.")
            return

        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime(self.TIMESTAMP_FORMAT)
        
        vocab_filename = f"{timestamp}{self.MODEL_FILE_SUFFIX}"
        vocab_path = os.path.join(path, vocab_filename)
        np.savez(vocab_path, vocabulary=self.vocabulary, k=np.array(self.k_value), 
                 label_map=self.label_map, classifier_type=self.classifier_type.name, 
                 descriptor_type=self.descriptor_type.name)
        print(f"Vocabulary saved to {vocab_path}")

        classifier_filename = f"{timestamp}{self.CLASSIFIER_FILE_SUFFIX}"
        classifier_path = os.path.join(path, classifier_filename)
        self.classifier.save(classifier_path)
        print(f"Classifier model saved to {classifier_path}")

    @staticmethod
    def load_model(vocab_path: str, classifier_path: str):
        try:
            data = np.load(vocab_path, allow_pickle=True)
            k_value = int(data['k'])
            k_node = next((kn for kn in KNodes if kn.value == k_value), None)
            if k_node is None: raise ValueError(f"K value {k_value} is not a valid KNodes member.")

            classifier_type = ClassifierType[data['classifier_type'].item()]
            descriptor_type = DescriptorType[data['descriptor_type'].item()]

            model = BagOfVisualWords(k=k_node, classifier_type=classifier_type, descriptor_type=descriptor_type)
            model.vocabulary = data['vocabulary']
            model.label_map = data['label_map'].item()

            if model.classifier_type == ClassifierType.SVM:
                model.classifier = cv2.ml.SVM_load(classifier_path)
            elif model.classifier_type == ClassifierType.LOGISTIC_REGRESSION:
                fs = cv2.FileStorage(classifier_path, cv2.FILE_STORAGE_READ)
                model.classifier.read(fs.getFirstTopLevelNode())
                fs.release()
            
            print(f"Model loaded successfully from {vocab_path} and {classifier_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
