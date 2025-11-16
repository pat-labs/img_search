import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List

import cv2
import numpy as np

from src.main.application.use_case.AnisotropicSIFT import AnisotropicSIFT
from src.main.application.use_case.ImageUtil import ImageDataFeature
from src.main.domain.ClassifierType import ClassifierType
from src.main.domain.DescriptorType import DescriptorType
from src.main.domain.Knodes import KNodes


class FisherMatrix:
    MAX_DESCRIPTORS_FOR_GMM = 1_000_000
    GMM_MAX_ITER = 100
    GMM_EPSILON = 0.1
    NORMALIZATION_EPSILON = 1e-6
    DESIRED_KEYPOINTS = 5_000
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

    def __init__(self, k: KNodes, descriptor_type: DescriptorType, classifier_type: ClassifierType):
        self.k = k
        self.descriptor_type = descriptor_type
        self.classifier_type = classifier_type
        
        self.gmm = None
        self.label_map = {}
        self.inverse_label_map = {}
        self.fisher_vectors = None

        self.classifier = self._create_classifier()
        self.descriptor = self._create_descriptor()
        self.descriptor_dimension = self.descriptor.descriptorSize()

    @staticmethod
    def compute_spatial_fisher_vector(image_data: ImageDataFeature, fm, grid_size=(2, 2)):
        """
        Computes a spatial pyramid of Fisher Vectors for an image.
        """
        h, w = image_data.shape
        keypoints, descriptors = image_data.keypoints, image_data.descriptor

        if descriptors is None or len(descriptors) == 0:
            # Return a zero vector if no features are found
            base_fv_dim = 2 * fm.k.value * fm.descriptor_dimension
            return np.zeros(base_fv_dim * grid_size[0] * grid_size[1], dtype=np.float32)

        # Assign descriptors to grid cells
        cell_descriptors = [[] for _ in range(grid_size[0] * grid_size[1])]
        cell_h, cell_w = h / grid_size[1], w / grid_size[0]

        for i, kp in enumerate(keypoints):
            x, y = kp
            col_idx = int(x // cell_w)
            row_idx = int(y // cell_h)
            cell_idx = row_idx * grid_size[0] + col_idx
            if 0 <= cell_idx < len(cell_descriptors):
                cell_descriptors[cell_idx].append(descriptors[i])

        # Compute Fisher Vector for each cell and concatenate
        final_spm_vector = []
        for cell_desc_list in cell_descriptors:
            if cell_desc_list:
                cell_desc_np = np.array(cell_desc_list, dtype=np.float32)
                fv_region = fm.compute_fisher_vector(cell_desc_np)
                final_spm_vector.append(fv_region)
            else:
                # Append a zero vector for empty cells
                base_fv_dim = 2 * fm.k.value * fm.descriptor_dimension
                final_spm_vector.append(np.zeros(base_fv_dim, dtype=np.float32))

        return np.concatenate(final_spm_vector)
        
    def is_trained(self) -> bool:
        return self.gmm is not None and self.classifier.isTrained()
        
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
        elif self.descriptor_type == DescriptorType.ANSIOTROPIC_SIFT:
            return AnisotropicSIFT()
        raise ValueError(f"Unsupported feature algorithm: {self.descriptor_type.name}")

    def _create_classifier(self):
        if self.classifier_type == ClassifierType.SVM:
            return cv2.ml.SVM_create()
        elif self.classifier_type == ClassifierType.LOGISTIC_REGRESSION:
            return cv2.ml.LogisticRegression_create()
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type.name}")

    def train(self, train_data: List[ImageDataFeature], parallel: bool = True):
        valid_desc = [item.descriptor for item in train_data if
                      item.descriptor is not None and len(item.descriptor) > 0]
        if not valid_desc:
            raise ValueError("No valid descriptors provided for GMM training.")

        all_descriptors = np.vstack(valid_desc).astype(np.float32)
        if len(all_descriptors) > self.MAX_DESCRIPTORS_FOR_GMM:
            np.random.shuffle(all_descriptors)
            all_descriptors = all_descriptors[:self.MAX_DESCRIPTORS_FOR_GMM]

        self._train_gmm(all_descriptors)

        if parallel:
            fisher_vectors, labels = self._compute_fisher_vectors_parallel(train_data)
        else:
            fisher_vectors, labels = self._compute_fisher_vectors_serial(train_data)

        if not fisher_vectors:
            raise ValueError("No valid Fisher vectors were generated.")

        self.fisher_vectors = np.vstack(fisher_vectors).astype(np.float32)

        unique_labels = sorted(list(set(labels)))
        self.label_map = {i: label for i, label in enumerate(unique_labels)}
        self.inverse_label_map = {label: i for i, label in enumerate(unique_labels)}
        numerical_labels = np.array([self.inverse_label_map[lbl] for lbl in labels], dtype=np.int32)

        self.classifier.train(self.fisher_vectors, cv2.ml.ROW_SAMPLE, numerical_labels)

    def _compute_fisher_vectors_serial(self, train_data: list[ImageDataFeature]):
        fisher_vectors, labels = [], []
        for item in train_data:
            fv = self.compute_fisher_vector(item.descriptor)
            if fv is not None and fv.size > 0:
                fisher_vectors.append(fv)
                labels.append(item.label)
        return fisher_vectors, labels

    def _compute_fisher_vectors_parallel(self, train_data: list[ImageDataFeature]):
        fisher_vectors, labels = [], []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.compute_fisher_vector, item.descriptor): item.label for item in train_data}
            for future in as_completed(futures):
                fv = future.result()
                if fv is not None and fv.size > 0:
                    fisher_vectors.append(fv)
                    labels.append(futures[future])
        return fisher_vectors, labels

    def predict(self, descriptor: np.ndarray) -> str | None:
        if not self.is_trained():
            return None

        fv = self.compute_fisher_vector(descriptor)
        if fv is None or fv.size == 0:
            return None

        fv = np.array([fv], dtype=np.float32)
        _, result = self.classifier.predict(fv)
        predicted_index = int(result[0][0])
        return self.label_map.get(predicted_index, "Unknown")

    def _train_gmm(self, descriptors: np.ndarray):
        self.gmm = cv2.ml.EM_create()
        self.gmm.setClustersNumber(self.k.value)
        self.gmm.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_DIAGONAL)
        term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.GMM_MAX_ITER, self.GMM_EPSILON)
        self.gmm.setTermCriteria(term_criteria)
        retval, _, _, _ = self.gmm.trainEM(descriptors)
        if not retval:
            raise RuntimeError("GMM training failed.")

    def compute_fisher_vector(self, descriptors: np.ndarray) -> np.ndarray:
        try:
            if descriptors is None or descriptors.size == 0:
                raise ValueError("Descriptor size error")

            means = self.gmm.getMeans()
            covs = self.gmm.getCovs()
            weights = self.gmm.getWeights()[0]

            if isinstance(covs, (tuple, list)):
                covs = np.array([np.diag(c) for c in covs], dtype=np.float32)

            if descriptors.shape[1] != means.shape[1]:
                raise ValueError("Descriptor means size error")

            posteriors = []
            for desc in descriptors:
                _, post = self.gmm.predict2(desc.reshape(1, -1))
                posteriors.append(post[0])
            posteriors = np.array(posteriors, dtype=np.float32)

            diff = descriptors[:, np.newaxis, :] - means[np.newaxis, :, :]
            stds = np.sqrt(covs[np.newaxis, :, :])

            mean_grad = np.sum(posteriors[:, :, np.newaxis] * diff / stds, axis=0)
            std_grad = np.sum(posteriors[:, :, np.newaxis] * ((diff ** 2 / covs[np.newaxis, :, :]) - 1), axis=0)

            fisher_vector = np.concatenate([mean_grad.flatten(), std_grad.flatten()])
            fisher_vector = np.sign(fisher_vector) * np.sqrt(np.abs(fisher_vector))
            norm = np.linalg.norm(fisher_vector)
            if norm > self.NORMALIZATION_EPSILON:
                fisher_vector /= norm

            return fisher_vector.astype(np.float32)

        except Exception:
            fisher_vector_dimension = 2 * self.k.value * self.descriptor_dimension
            return np.zeros(fisher_vector_dimension, dtype=np.float32)

    def save_model(self, directory_path: str, file_name: str = None) -> str | None:
        if not self.is_trained():
            print("Model has not been trained. Cannot save.")
            return None

        os.makedirs(directory_path, exist_ok=True)

        if file_name is None:
            timestamp = datetime.now().strftime(self.TIMESTAMP_FORMAT)
            file_name = f"fisher_{timestamp}"

        gmm_path = os.path.join(directory_path, f"{file_name}_gmm.xml")
        classifier_path = os.path.join(directory_path, f"{file_name}_classifier.xml")
        full_path = os.path.join(directory_path, f"{file_name}.pkl")

        try:
            gmm_fs = cv2.FileStorage(gmm_path, cv2.FileStorage_WRITE)
            self.gmm.write(gmm_fs)
            gmm_fs.release()
            print(f"💾 GMM saved to {gmm_path}")

            self.classifier.save(classifier_path)
            print(f"💾 Classifier saved to {classifier_path}")
        except Exception as e:
            print(f"❌ Failed to save GMM or classifier: {e}")
            return None

        state = {
            "k": self.k,
            "descriptor_type": self.descriptor_type,
            "classifier_type": self.classifier_type,
            "label_map": self.label_map,
            "inverse_label_map": self.inverse_label_map,
        }

        try:
            with open(full_path, "wb") as f:
                pickle.dump(state, f)
            print(f"💾 Model (metadata) saved to {full_path}")
        except Exception as e:
            print(f"❌ Failed to save main model file: {e}")
            return None

        return full_path

    @staticmethod
    def load_model(model_path: str):
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None

        directory_path = os.path.dirname(model_path)
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        gmm_path = os.path.join(directory_path, f"{base_name}_gmm.xml")
        classifier_path = os.path.join(directory_path, f"{base_name}_classifier.xml")

        try:
            with open(model_path, "rb") as f:
                state = pickle.load(f)

            model = FisherMatrix(
                state["k"],
                state["descriptor_type"],
                state["classifier_type"]
            )

            model.label_map = state["label_map"]
            model.inverse_label_map = state["inverse_label_map"]

            model.gmm = cv2.ml.EM_create()
            fs_gmm = cv2.FileStorage(gmm_path, cv2.FileStorage_READ)
            model.gmm.read(fs_gmm.root())
            fs_gmm.release()
            print(f"📂 GMM loaded from {gmm_path}")

            if state["classifier_type"].name == "SVM":
                model.classifier = cv2.ml.SVM_load(classifier_path)
            elif state["classifier_type"].name == "LOGISTIC_REGRESSION":
                model.classifier = cv2.ml.LogisticRegression_load(classifier_path)
            else:
                raise ValueError(f"Unsupported classifier type: {state['classifier_type'].name}")

            return model

        except Exception as e:
            print(f"❌ Failed to load FisherMatrix model: {e}")
            return None
