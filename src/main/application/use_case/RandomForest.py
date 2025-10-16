import cv2
import numpy as np
import os
import pickle
from collections import Counter

from src.main.application.use_case.ImageDescriptorAnalyzer import DescriptorType
from src.main.application.use_case.ImageUtil import ImageUtil, PathLabel


class RandomForest:

    def __init__(self, descriptor_type: DescriptorType = DescriptorType.SIFT):
        self.descriptor_type = descriptor_type
        self.descriptor = self.descriptor_type.create_descriptor()
        self.classifier = cv2.ml.RTrees_create()
        self.label_map = None

    def is_trained(self) -> bool:
        return self.classifier.isTrained()

    def train(self, train_data: list[PathLabel]):
        print(f"Starting Random Forest training with {self.descriptor_type.name} descriptors...")

        all_descriptors = []
        all_labels = []

        # Create a training set where each descriptor is a sample
        for item in train_data:
            try:
                img = ImageUtil.load_grayscale_image(item.path)
                _, descriptors = self.descriptor.detectAndCompute(img, None)
                if descriptors is not None:
                    all_descriptors.extend(descriptors)
                    # Assign the image's label to each of its descriptors
                    all_labels.extend([item.label] * len(descriptors))
            except Exception as e:
                print(f"Error processing {item.path}: {e}")

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

    def predict(self, image_path: str) -> str | None:
        if not self.is_trained():
            print("Prediction failed: The model has not been trained yet.")
            return None

        img = ImageUtil.load_grayscale_image(image_path)
        _, descriptors = self.descriptor.detectAndCompute(img, None)

        if descriptors is None or len(descriptors) == 0:
            return "Unknown"  # Cannot make a prediction

        # Predict the class for each descriptor and find the most common one (voting)
        _, results = self.classifier.predict(descriptors.astype(np.float32))
        predicted_indices = [int(r[0]) for r in results]

        if not predicted_indices:
            return "Unknown"

        most_common_index = Counter(predicted_indices).most_common(1)[0][0]
        return self.label_map.get(most_common_index, "Unknown")

    def save_model(self, path: str):
        if not self.is_trained():
            print("Model has not been trained. Cannot save.")
            return

        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"RandomForestImageClassifier model saved to {path}")

    def __getstate__(self):
        """Prepare the object for pickling, handling non-serializable attributes."""
        state = self.__dict__.copy()

        # Remove the non-pickleable descriptor object
        del state['descriptor']

        # Manually serialize the OpenCV classifier to a memory buffer
        if state['classifier'] is not None and state['classifier'].isTrained():
            fs = cv2.FileStorage(".xml", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
            state['classifier'].write(fs)
            state['classifier'] = fs.release()
        else:
            state['classifier'] = None

        return state

    def __setstate__(self, state):
        """Restore the object after unpickling."""
        self.__dict__.update(state)

        # Recreate the non-serializable attributes
        self.descriptor = self.descriptor_type.create_descriptor()
        if self.classifier is not None:
            # First, create the classifier object
            self.classifier = cv2.ml.RTrees_create()
            # Then, load the data into it in-place
            self.classifier.read(cv2.FileStorage(state['classifier'], cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY).getFirstTopLevelNode())
        else:
            # If no classifier was saved, create a new empty one
            self.classifier = cv2.ml.RTrees_create()

    @staticmethod
    def load_model(path: str):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"RandomForestImageClassifier model loaded from {path}")
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return None
