import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from src.main.application.use_case.ImageDescriptorAnalyzer import ImageDescriptorAnalyzer, DescriptorType
from src.main.application.use_case.FisherMatrix import FisherMatrix, KNodes
from src.main.application.use_case.ImageUtil import ImageUtil, PathLabel
from src.main.application.use_case.ClassifierType import ClassifierType

class TestFisherMatrix(unittest.TestCase):

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

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_train_predict_and_save(self):
        # 1. Initialize and train the model
        fm = FisherMatrix(
            k=KNodes.K16,
            descriptor_type=self.descriptor_type,
            classifier_type=self.classifier_type
        )
        train_data = ImageUtil.load_image_paths_and_labels(self.dataset_dir)
        fm.train(train_data, self.all_descriptors)

        # 2. Assert that the model is trained
        self.assertTrue(fm.is_trained())

        # 3. Test prediction on a training image
        test_image_path = os.path.join(self.class_a_path, "a_0.png")
        prediction = fm.predict(test_image_path)
        self.assertEqual(prediction, "class_a")

        # 4. Save the model
        model_path = os.path.join(self.model_dir, "fisher_matrix.pkl")
        fm.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))

        fm_loaded = FisherMatrix.load_model(model_path)

        # 3. Assert that the loaded model is valid and can predict
        self.assertIsNotNone(fm_loaded)
        self.assertTrue(fm_loaded.is_trained())
        test_image_path = os.path.join(self.class_b_path, "b_1.png")
        prediction = fm_loaded.predict(test_image_path)
        self.assertEqual(prediction, "class_b")


if __name__ == '__main__':
    unittest.main()