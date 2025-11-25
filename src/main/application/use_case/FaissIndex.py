from enum import Enum, auto

import faiss
import numpy as np


class FaissMetric(Enum):
    L2 = auto()
    COSINE = auto()
    INNER_PRODUCT = auto()


class FaissIndex:
    def __init__(self, metric: FaissMetric = FaissMetric.L2):
        self.metric = metric
        self.index = None
        self.metadata = None

    def build_index(self, fisher_matrix: np.ndarray, metadata: list[str] | None = None):
        if fisher_matrix is None or fisher_matrix.shape[0] == 0:
            print("Cannot build index from empty data.")
            return

        vectors = fisher_matrix.astype('float32')
        faiss.normalize_L2(vectors)

        if self.metric == FaissMetric.COSINE:
            faiss.normalize_L2(vectors)
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        elif self.metric == FaissMetric.INNER_PRODUCT:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        else:
            self.index = faiss.IndexFlatL2(vectors.shape[1])

        print(
            f"Building Faiss index ({self.metric.name}) with {vectors.shape[0]} vectors of dimension {vectors.shape[1]}...")
        self.index.add(vectors)
        print("Faiss index built successfully.")

        if metadata is not None:
            if len(metadata) != vectors.shape[0]:
                print("Warning: metadata length does not match number of vectors.")
            self.metadata = metadata
        else:
            self.metadata = [str(i) for i in range(vectors.shape[0])]

    def save_index(self, path: str, metadata_path: str | None = None):
        if self.index is None:
            print("Index has not been built. Cannot save.")
            return
        print(f"Saving Faiss index to {path}...")
        faiss.write_index(self.index, path)
        print("Index saved successfully.")
        if metadata_path and self.metadata is not None:
            np.save(metadata_path, np.array(self.metadata, dtype=object))
            print(f"Metadata saved to {metadata_path}")

    def load_index(self, path: str, metadata_path: str | None = None):
        print(f"Loading Faiss index from {path}...")
        self.index = faiss.read_index(path)
        print("Index loaded successfully.")
        if metadata_path:
            try:
                self.metadata = np.load(metadata_path, allow_pickle=True).tolist()
                print(f"Metadata loaded from {metadata_path}")
            except Exception as e:
                print(f"Failed to load metadata: {e}")
                self.metadata = None

    def search(self, query_vector: np.ndarray, max_items: int | None = None):
        if self.index is None:
            raise RuntimeError("Index is not built or loaded. Cannot perform search.")
        if max_items is None:
            max_items = self.index.ntotal

        query = np.array([query_vector], dtype='float32')
        if query.shape[1] != self.index.d:
            raise ValueError(f"Query dimension {query.shape[1]} does not match index dimension {self.index.d}")
        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, max_items)
        results = []
        for i, d in zip(indices[0], distances[0]):
            # Only process and return valid indices. Ignore -1 placeholders from Faiss.
            if i != -1 and self.metadata is not None and 0 <= i < len(self.metadata):
                results.append((self.metadata[i], float(d)))
        return results

    def get_distance(self, metadata_item1: str, metadata_item2: str) -> float:
        """
        Calculates the distance between two items in the index, identified by their metadata.
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index or metadata not loaded.")

        try:
            idx1 = self.metadata.index(metadata_item1)
            idx2 = self.metadata.index(metadata_item2)
        except ValueError as e:
            raise ValueError(f"One of the metadata items not found in index: {e}")

        # Reconstruct returns the original, un-normalized vectors.
        vec1 = self.index.reconstruct(idx1).reshape(1, -1)
        vec2 = self.index.reconstruct(idx2).reshape(1, -1)

        # We must normalize them just like during search to get the correct distance.
        faiss.normalize_L2(vec1)
        faiss.normalize_L2(vec2)

        # For L2, distance is the norm of the difference.
        return float(np.linalg.norm(vec1 - vec2))
