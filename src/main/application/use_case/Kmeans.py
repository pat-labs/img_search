import os
import pickle
from datetime import datetime
from typing import List

import numpy as np
import cv2

from src.main.application.use_case.ImageUtil import ImageDataFeature
from src.main.domain.Knodes import KNodes
from src.main.presentation.ImageDescriptorAnalyzer import DescriptorType


class KMeans:
    MAX_ITER = 300
    TOLERANCE = 1e-4
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

    def __init__(self, k: KNodes, descriptor_type: DescriptorType, max_iter=None, tol=None):
        self.k = k
        self.descriptor_type = descriptor_type
        self.max_iter = max_iter if max_iter is not None else self.MAX_ITER
        self.tol = tol if tol is not None else self.TOLERANCE

        self.descriptor = self.descriptor_type.create_descriptor()
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

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

    def fit(self, train_data: List[ImageDataFeature], parallel: bool = True):
        valid_desc = [item.descriptor for item in train_data if
                      item.descriptor is not None and len(item.descriptor) > 0]
        if not valid_desc:
            raise ValueError("No valid descriptors provided for GMM training.")

        X = np.vstack(valid_desc).astype(np.float32)

        n_samples, _ = X.shape
        random_indices = np.random.choice(n_samples, self.k.value, replace=False)
        self.cluster_centers_ = X[random_indices].copy()

        for i in range(self.max_iter):
            labels = self._assign_labels(X, parallel)
            new_cluster_centers = self._update_centers(X, labels)
            if np.all(np.linalg.norm(new_cluster_centers - self.cluster_centers_, axis=1) < self.tol):
                break
            self.cluster_centers_ = new_cluster_centers

        self.labels_ = self._assign_labels(X)
        self._calculate_inertia(X)
        print("Fit complete.")
        return self
    
    def _assign_labels(self, X, parallel: bool = True):
        # This vectorized operation is highly efficient and leverages NumPy's C backend,
        # which is often multi-threaded, effectively making it a parallel operation.
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def _update_centers(self, X, labels):
        n_features = X.shape[1]
        new_centers = np.zeros((self.k.value, n_features))
        counts = np.zeros(self.k.value)
        for i in range(X.shape[0]):
            cluster_idx = labels[i]
            counts[cluster_idx] += 1
            new_centers[cluster_idx] += X[i]
        for i in range(self.k.value):
            if counts[i] > 0:
                new_centers[i] /= counts[i]
        return new_centers

    def _calculate_inertia(self, X):
        inertia = 0.0
        for i, label in enumerate(self.labels_):
            inertia += np.linalg.norm(X[i] - self.cluster_centers_[label]) ** 2
        self.inertia_ = inertia
        print(f"Inertia: {self.inertia_:.2f}")

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("The model has not been trained yet. Call fit() or train() first.")
        return self._assign_labels(X)
    
    def save_model(self, directory_path: str, base_filename: str = None) -> str | None:
        if self.cluster_centers_ is None:
            print("Model has not been trained. Cannot save.")
            return None

        os.makedirs(directory_path, exist_ok=True)

        if base_filename is None:
            timestamp = datetime.now().strftime(self.TIMESTAMP_FORMAT)
            base_filename = f"kmeans_{timestamp}"

        # Define paths for components
        centers_path = os.path.join(directory_path, f"{base_filename}_centers.npz")
        main_model_path = os.path.join(directory_path, f"{base_filename}.pkl")

        # Save cluster centers to their own file
        np.savez(centers_path, cluster_centers_=self.cluster_centers_)
        print(f"KMeans centers saved to {centers_path}")

        state = {
            "k": self.k,
            "descriptor_type": self.descriptor_type,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "labels_": self.labels_,
            "inertia_": self.inertia_
        }

        try:
            with open(main_model_path, "wb") as f:
                pickle.dump(state, f)
            print(f"✅ KMeans model saved to {main_model_path}")
            return main_model_path
        except Exception as e:
            print(f"❌ Failed to save main KMeans model file: {e}")
            return None

    @staticmethod
    def load_model(model_path: str):
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None

        directory_path = os.path.dirname(model_path)
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        centers_path = os.path.join(directory_path, f"{base_name}_centers.npz")

        try:
            with open(model_path, 'rb') as f:
                state = pickle.load(f)

            model = KMeans(
                k=state["k"],
                descriptor_type=state["descriptor_type"],
                max_iter=state["max_iter"],
                tol=state["tol"]
            )

            model.labels_ = state["labels_"]
            model.inertia_ = state["inertia_"]

            if os.path.exists(centers_path):
                model.cluster_centers_ = np.load(centers_path)['cluster_centers_']
                print(f"📂 KMeans centers loaded from {centers_path}")
            else:
                raise FileNotFoundError(f"KMeans centers file not found at {centers_path}")

            print(f"✅ KMeans model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"❌ Failed to load KMeans model: {e}")
            return None
