import cv2
import numpy as np
from enum import Enum

from src.main.application.use_case.ImageUtil import PathLabel, ImageUtil


class KNodes(Enum):
    K16 = 16
    K32 = 32
    K64 = 64
    K128 = 128
    K256 = 256


class FisherMatrix:
    SIFT_DESCRIPTOR_DIMENSION = 128
    MAX_DESCRIPTORS_FOR_GMM = 1_000_000
    GMM_MAX_ITER = 100
    GMM_EPSILON = 0.1
    NORMALIZATION_EPSILON = 1e-6

    def __init__(self, k: KNodes):
        self.k = k.value
        self.gmm = None
        self.sift = cv2.SIFT_create()

    def create(self, train_data: PathLabel):
        descriptors = self._get_descriptors(train_data)
        if descriptors is not None:
            prepped_descriptors = self._prepare_descriptors_for_gmm(descriptors)
            self._train_gmm(prepped_descriptors)

    def _get_descriptors(self, data: PathLabel):
        print("Extracting SIFT features...")
        all_descriptors = []
        for i, (image_path, _) in enumerate(data):
            try:
                img = ImageUtil.load_grayscale_image(image_path)
                _, descriptors = self.sift.detectAndCompute(img, None)
                if descriptors is not None:
                    all_descriptors.append(descriptors)
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(data)} images")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        if not all_descriptors:
            print("No descriptors found.")
            return None
        return all_descriptors

    def _prepare_descriptors_for_gmm(self, descriptors):
        print("Stacking and sampling descriptors...")
        all_descriptors = np.vstack(descriptors)
        if len(all_descriptors) > self.MAX_DESCRIPTORS_FOR_GMM:
            print(f"Found {len(all_descriptors)} descriptors, using a random subset of {self.MAX_DESCRIPTORS_FOR_GMM}.")
            np.random.shuffle(all_descriptors)
            all_descriptors = all_descriptors[:self.MAX_DESCRIPTORS_FOR_GMM]
        return all_descriptors.astype(np.float32)

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

    def evaluate_model(self, test_data: PathLabel):
        if self.gmm is None:
            print("GMM model is not trained. Please call create() first.")
            return None, None

        print(f"Evaluating model on {len(test_data)} test images...")
        fisher_vectors, labels = [], []
        for i, (image_path, label) in enumerate(test_data):
            fv = self._process_single_image_for_evaluation(image_path)
            fisher_vectors.append(fv)
            labels.append(label)
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(test_data)} images")

        print("Evaluation complete.")
        return np.array(fisher_vectors), np.array(labels)

    def _process_single_image_for_evaluation(self, image_path: str):
        try:
            img = ImageUtil.load_grayscale_image(image_path)
            _, descriptors = self.sift.detectAndCompute(img, None)

            if descriptors is None or len(descriptors) == 0:
                print(f"Warning: No descriptors found for {image_path}")
                fisher_vector_dimension = 2 * self.k * self.SIFT_DESCRIPTOR_DIMENSION
                return np.zeros(fisher_vector_dimension, dtype=np.float32)
            
            return self._compute_fisher_vector(descriptors.astype(np.float32))

        except Exception as e:
            print(f"Error evaluating image {image_path}: {e}")
            fisher_vector_dimension = 2 * self.k * self.SIFT_DESCRIPTOR_DIMENSION
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
        
        descriptors_r = descriptors[:, np.newaxis, :]
        means_r = means[np.newaxis, :, :]
        std_devs_r = std_devs[np.newaxis, :, :]
        posteriors_r = posteriors[:, :, np.newaxis]

        diff = descriptors_r - means_r
        normalized_diff = diff / std_devs_r
        weighted_diff = posteriors_r * normalized_diff
        
        sum_over_descriptors = np.sum(weighted_diff, axis=0)
        
        weights_r = weights[:, np.newaxis]
        normalization_factor = num_descriptors * np.sqrt(weights_r)
        
        mean_gradient = sum_over_descriptors / normalization_factor
        return mean_gradient

    def _calculate_std_dev_gradient(self, descriptors, means, std_devs, weights, posteriors):
        num_descriptors = descriptors.shape[0]

        descriptors_r = descriptors[:, np.newaxis, :]
        means_r = means[np.newaxis, :, :]
        std_devs_r = std_devs[np.newaxis, :, :]
        posteriors_r = posteriors[:, :, np.newaxis]

        diff = descriptors_r - means_r
        normalized_term = ((diff / std_devs_r) ** 2) - 1
        weighted_term = posteriors_r * normalized_term
        
        sum_over_descriptors = np.sum(weighted_term, axis=0)
        
        weights_r = weights[:, np.newaxis]
        normalization_factor = num_descriptors * np.sqrt(2 * weights_r)
        
        std_dev_gradient = sum_over_descriptors / normalization_factor
        return std_dev_gradient

    def _normalize_fisher_vector(self, fisher_vector):
        power_normalized_fv = np.sign(fisher_vector) * np.sqrt(np.abs(fisher_vector))
        
        norm = np.linalg.norm(power_normalized_fv)
        if norm > self.NORMALIZATION_EPSILON:
            l2_normalized_fv = power_normalized_fv / norm
        else:
            l2_normalized_fv = power_normalized_fv
            
        return l2_normalized_fv
