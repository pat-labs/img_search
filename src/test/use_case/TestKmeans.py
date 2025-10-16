import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from src.main.application.use_case.ImageDescriptorAnalyzer import ImageDescriptorAnalyzer, DescriptorType
from src.main.application.use_case.Kmeans import KMeans
from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.application.use_case.ClassifierType import ClassifierType


class TestKmeans(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.test_dir, "model")
        self.dataset_dir = os.path.join(self.test_dir, "dataset")
        os.makedirs(self.model_dir)

        # Create a dummy dataset with two classes
        self.class_a_path = os.path.join(self.dataset_dir, "class_a")
        self.class_b_path = os.path.join(self.dataset_dir, "class_b")
        os.makedirs(self.class_a_path)
        os.makedirs(self.class_b_path)

        # Create a few dummy images with simple shapes for reliable feature detection by ORB/SIFT
        for i in range(3):
            # Image A: White square on black background
            img_a = np.zeros((50, 50), dtype=np.uint8)
            cv2.rectangle(img_a, (10, 10 + i), (25, 25 + i), 255,
                          -1)  # A filled rectangle, slightly different each time

            # Image B: White circle on black background
            img_b = np.zeros((50, 50), dtype=np.uint8)
            cv2.circle(img_b, (25, 25), 10 + i, 255, -1)  # A filled circle, slightly different each time
            cv2.imwrite(os.path.join(self.class_a_path, f"a_{i}.png"), img_a)
            cv2.imwrite(os.path.join(self.class_b_path, f"b_{i}.png"), img_b)

        self.descriptor_type = DescriptorType.SIFT
        self.classifier_type = ClassifierType.SVM
        path_and_labels = ImageUtil.load_image_paths_and_labels(self.dataset_dir)
        self.all_descriptors = ImageDescriptorAnalyzer.extract_features_serial([item.path for item in path_and_labels],
                                                                               self.descriptor_type)
        self.n_clusters = 4

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_fit_predict_and_save(self):
        # 1. Initialize and fit the model
        kmeans = KMeans(n_clusters=self.n_clusters, descriptor_type=self.descriptor_type)

        kmeans.fit(self.all_descriptors)

        # 2. Assert that the model is trained
        self.assertIsNotNone(kmeans.cluster_centers_)
        self.assertEqual(kmeans.cluster_centers_.shape, (self.n_clusters, 128)) # SIFT descriptors are 128-dim
        self.assertIsNotNone(kmeans.labels_)
        self.assertGreater(kmeans.inertia_, 0)

        # 3. Test prediction
        # Use a subset of the original descriptors for prediction
        test_descriptors = self.all_descriptors[0] 
        predictions = kmeans.predict(test_descriptors)
        self.assertEqual(len(predictions), len(test_descriptors))
        self.assertTrue(all(0 <= p < self.n_clusters for p in predictions))

        # 4. Save the model
        model_path = os.path.join(self.model_dir, "kmeans.pkl")
        kmeans.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))

        # 2. Load the model
        kmeans_loaded = KMeans.load_model(model_path)

        # 3. Assert that the loaded model is valid and has the correct state
        self.assertIsNotNone(kmeans_loaded)
        self.assertEqual(kmeans_loaded.n_clusters, self.n_clusters)
        self.assertEqual(kmeans_loaded.descriptor_type, self.descriptor_type)
        self.assertIsNotNone(kmeans_loaded.cluster_centers_)
        # Compare the loaded cluster centers to the original ones
        np.testing.assert_array_equal(kmeans.cluster_centers_, kmeans_loaded.cluster_centers_)

if __name__ == '__main__':
    unittest.main()