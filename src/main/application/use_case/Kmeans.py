import cv2
import numpy as np
from typing import List
import pickle
import os

from src.main.application.use_case.ImageDescriptorAnalyzer import DescriptorType
from src.main.application.use_case.ImageUtil import PathLabel, ImageUtil


class KMeans:
    MAX_ITER = 300
    TOLERANCE = 1e-4

    def __init__(self, n_clusters, descriptor_type: DescriptorType = DescriptorType.SIFT, max_iter=None, tol=None):
        self.n_clusters = n_clusters
        self.descriptor_type = descriptor_type
        self.max_iter = max_iter if max_iter is not None else self.MAX_ITER
        self.tol = tol if tol is not None else self.TOLERANCE
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, precomputed_descriptors: list[np.ndarray]):
        print(f"Fitting KMeans with {self.descriptor_type.name} descriptors...")
        if precomputed_descriptors is None or len(precomputed_descriptors) == 0:
            print("Fit failed: No features could be extracted.")
            return self

        # Stack the list of descriptor arrays into a single NumPy array for clustering.
        # Also, ensure the data type is float32 for distance calculations.
        X = np.vstack(precomputed_descriptors).astype(np.float32)

        n_samples, _ = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices].copy()

        for i in range(self.max_iter):
            labels = self._assign_labels(X)
            new_cluster_centers = self._update_centers(X, labels)
            if np.all(np.linalg.norm(new_cluster_centers - self.cluster_centers_, axis=1) < self.tol):
                break
            self.cluster_centers_ = new_cluster_centers

        self.labels_ = self._assign_labels(X)
        self._calculate_inertia(X)
        print("Fit complete.")
        return self

    def save_model(self, path: str):
        if self.cluster_centers_ is None:
            print("Model has not been trained. Cannot save.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"KMeans model saved to {path}")

    @staticmethod
    def load_model(path: str):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"KMeans model loaded from {path}")
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return None

    def _assign_labels(self, X):
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - self.cluster_centers_, axis=1)
            labels[i] = np.argmin(distances)
        return labels

    def _update_centers(self, X, labels):
        n_features = X.shape[1]
        new_centers = np.zeros((self.n_clusters, n_features))
        counts = np.zeros(self.n_clusters)
        for i in range(X.shape[0]):
            cluster_idx = labels[i]
            counts[cluster_idx] += 1
            new_centers[cluster_idx] += X[i]
        for i in range(self.n_clusters):
            if counts[i] > 0:
                new_centers[i] /= counts[i]
        return new_centers

    def _calculate_inertia(self, X):
        inertia = 0.0
        for i, label in enumerate(self.labels_):
            inertia += np.linalg.norm(X[i] - self.cluster_centers_[label])**2
        self.inertia_ = inertia
        print(f"Inertia: {self.inertia_:.2f}")

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("The model has not been trained yet. Call fit() or train() first.")
        return self._assign_labels(X)
