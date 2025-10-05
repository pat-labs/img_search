import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        n_samples, n_features = X.shape

        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices].copy()

        for i in range(self.max_iter):
            labels = self._assign_labels(X)

            new_cluster_centers = self._update_centers(X, labels)

            if np.all(np.linalg.norm(new_cluster_centers - self.cluster_centers_, axis=1) < self.tol):
                break

            self.cluster_centers_ = new_cluster_centers

        self.labels_ = self._assign_labels(X)
        return self

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

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("The model has not been trained yet. Call fit() first.")

        return self._assign_labels(X)
