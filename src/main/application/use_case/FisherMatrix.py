import cv2
import numpy as np
from enum import Enum
import pickle
import os

from src.main.application.use_case.ClassifierType import ClassifierType
from src.main.application.use_case.ImageDescriptorAnalyzer import DescriptorType
from src.main.application.use_case.ImageUtil import PathLabel, ImageUtil
from io import BytesIO

class KNodes(Enum):
    K16 = 16
    K32 = 32
    K64 = 64
    K128 = 128
    K256 = 256

class FisherMatrix:
    MAX_DESCRIPTORS_FOR_GMM = 1_000_000
    GMM_MAX_ITER = 100
    GMM_EPSILON = 0.1
    NORMALIZATION_EPSILON = 1e-6

    def __init__(self, k: KNodes, 
                 descriptor_type: DescriptorType, 
                 classifier_type: ClassifierType):
        self.k = k.value
        self.descriptor_type = descriptor_type
        self.classifier_type = classifier_type
        self.gmm = None
        self.classifier = self.classifier_type.create_classifier()
        self.label_map = None
        self.descriptor = self.descriptor_type.create_descriptor()
        self.descriptor_dimension = self.descriptor.descriptorSize()

    def train(self, train_data: list[PathLabel], precomputed_descriptors: list[np.ndarray]):
        print(f"Starting training with {self.descriptor_type.name} and {self.classifier_type.name}...")
        all_descriptors = np.vstack(precomputed_descriptors)
        if len(all_descriptors) > self.MAX_DESCRIPTORS_FOR_GMM:
            print(f"Found {len(all_descriptors)} descriptors, using a random subset of {self.MAX_DESCRIPTORS_FOR_GMM}.")
            np.random.shuffle(all_descriptors)
            all_descriptors = all_descriptors[:self.MAX_DESCRIPTORS_FOR_GMM]
        self._train_gmm(all_descriptors)

        print("Encoding training data into Fisher Vectors...")
        fisher_vectors, labels = self.encode(train_data)
        if fisher_vectors is None or len(fisher_vectors) == 0:
            print("Classifier training failed: Could not encode training data.")
            return

        print(f"Training the {self.classifier_type.name} classifier...")
        label_map = {label: i for i, label in enumerate(np.unique(labels))}
        self.label_map = {i: label for label, i in label_map.items()}
        numerical_labels = np.array([label_map[label] for label in labels], dtype=np.int32)

        self.classifier.train(fisher_vectors, cv2.ml.ROW_SAMPLE, numerical_labels)
        print("Training complete.")

    def predict(self, image_path: str) -> str | None:
        if not self.is_trained():
            print("Prediction failed: The model has not been trained yet.")
            return None
        
        fv = self._process_single_image(image_path)
        fv = np.array([fv], dtype=np.float32)

        _, result = self.classifier.predict(fv)
        predicted_label_index = int(result[0][0])
        
        return self.label_map.get(predicted_label_index, "Unknown")

    def is_trained(self) -> bool:
        return self.gmm is not None and self.classifier.isTrained()

    def encode(self, image_data: list[PathLabel]) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.gmm is None:
            print("Model is not trained. Please call train() first.")
            return None, None

        fisher_vectors, labels = [], []
        for item in image_data:
            fv = self._process_single_image(item.path)
            fisher_vectors.append(fv)
            labels.append(item.label)

        return np.array(fisher_vectors), np.array(labels)

    def save_model(self, path: str):
        if not self.is_trained():
            print("Model has not been trained. Cannot save.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"FisherMatrix model saved to {path}")

    @staticmethod
    def load_model(path: str):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"FisherMatrix model loaded from {path}")
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return None

    def __getstate__(self):
        """Prepare the object for pickling, handling non-serializable attributes."""
        state = self.__dict__.copy()
        
        # Remove the non-pickleable descriptor object
        del state['descriptor']

        # Serialize GMM and Classifier manually
        if state['gmm'] is not None:
            fs = cv2.FileStorage(".xml", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
            state['gmm'].write(fs)
            state['gmm'] = fs.release()

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
        
        # Recreate the descriptor object
        self.descriptor = self.descriptor_type.create_descriptor()
        self.gmm = cv2.ml.EM_create().read(cv2.FileStorage(self.gmm, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY).getFirstTopLevelNode())
        self.classifier = self.classifier_type.create_classifier().read(cv2.FileStorage(self.classifier, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY).getFirstTopLevelNode())

    def _train_gmm(self, descriptors):
        print(f"Training GMM with {self.k} components...")
        self.gmm = cv2.ml.EM_create()
        self.gmm.setClustersNumber(self.k)
        self.gmm.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_DIAGONAL)
        term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.GMM_MAX_ITER, self.GMM_EPSILON)
        self.gmm.setTermCriteria(term_criteria)
        
        retval, _, _, _ = self.gmm.trainEM(descriptors)
        if retval:
            print("GMM trained successfully.")
        else:
            print("GMM training failed.")

    def _process_single_image(self, image_path: str):
        try:
            img = ImageUtil.load_grayscale_image(image_path)
            _, descriptors = self.descriptor.detectAndCompute(img, None)

            if descriptors is None or len(descriptors) == 0:
                fisher_vector_dimension = 2 * self.k * self.descriptor_dimension
                return np.zeros(fisher_vector_dimension, dtype=np.float32)
            
            return self._compute_fisher_vector(descriptors.astype(np.float32))
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            fisher_vector_dimension = 2 * self.k * self.descriptor_dimension
            return np.zeros(fisher_vector_dimension, dtype=np.float32)

    def _compute_fisher_vector(self, descriptors):
        means, std_devs, weights = self._get_gmm_parameters()
        posteriors = self._calculate_posteriors(descriptors)

        grad_mean = self._calculate_mean_gradient(descriptors, means, std_devs, weights, posteriors)
        grad_std = self._calculate_std_dev_gradient(descriptors, means, std_devs, weights, posteriors)

        fisher_vector = np.concatenate((grad_mean.flatten(), grad_std.flatten()))
        return self._normalize_fisher_vector(fisher_vector)

    def _get_gmm_parameters(self):
        means = self.gmm.getMeans()
        covs = self.gmm.getCovs()
        weights = self.gmm.getWeights()[0]
        variances = np.array([np.diag(c) for c in covs])
        std_devs = np.sqrt(variances)
        return means, std_devs, weights

    def _calculate_posteriors(self, descriptors):
        _, posteriors = self.gmm.predict2(descriptors)
        return posteriors

    def _calculate_mean_gradient(self, descriptors, means, std_devs, weights, posteriors):
        num_descriptors = descriptors.shape[0]
        diff = descriptors[:, np.newaxis, :] - means[np.newaxis, :, :]
        normalized_diff = diff / std_devs[np.newaxis, :, :]
        weighted_diff = posteriors[:, :, np.newaxis] * normalized_diff
        sum_over_descriptors = np.sum(weighted_diff, axis=0)
        mean_gradient = sum_over_descriptors / (num_descriptors * np.sqrt(weights[:, np.newaxis]))
        return mean_gradient

    def _calculate_std_dev_gradient(self, descriptors, means, std_devs, weights, posteriors):
        num_descriptors = descriptors.shape[0]
        diff_sq = ((descriptors[:, np.newaxis, :] - means[np.newaxis, :, :]) / std_devs[np.newaxis, :, :]) ** 2 - 1
        weighted_term = posteriors[:, :, np.newaxis] * diff_sq
        sum_over_descriptors = np.sum(weighted_term, axis=0)
        std_dev_gradient = sum_over_descriptors / (num_descriptors * np.sqrt(2 * weights[:, np.newaxis]))
        return std_dev_gradient

    def _normalize_fisher_vector(self, fisher_vector):
        power_normalized_fv = np.sign(fisher_vector) * np.sqrt(np.abs(fisher_vector))
        norm = np.linalg.norm(power_normalized_fv)
        if norm > self.NORMALIZATION_EPSILON:
            l2_normalized_fv = power_normalized_fv / norm
        else:
            l2_normalized_fv = power_normalized_fv
        return l2_normalized_fv
