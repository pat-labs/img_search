import unittest
import os
import tempfile
import shutil
import numpy as np
import cv2
import multiprocessing

from src.main.application.use_case.RandomForest import RandomForest
from src.main.application.use_case.ImageDescriptorAnalyzer import DescriptorType
from src.main.application.use_case.ImageUtil import ImageUtil

class TestRandomForest(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "rf_model.pkl")
        self.dataset_dir = os.path.join(self.test_dir, "dataset")
        os.makedirs(self.dataset_dir)

        self.class_a_path = os.path.join(self.dataset_dir, "class_a")
        self.class_b_path = os.path.join(self.dataset_dir, "class_b")
        os.makedirs(self.class_a_path)
        os.makedirs(self.class_b_path)

        for i in range(3):
            # Image A: White square on black background
            img_a = np.zeros((50, 50), dtype=np.uint8)
            cv2.rectangle(img_a, (10, 10 + i), (25, 25 + i), 255, -1)
            
            # Image B: White circle on black background
            img_b = np.zeros((50, 50), dtype=np.uint8)
            cv2.circle(img_b, (25, 25), 10 + i, 255, -1)
            cv2.imwrite(os.path.join(self.class_a_path, f"a_{i}.png"), img_a)
            cv2.imwrite(os.path.join(self.class_b_path, f"b_{i}.png"), img_b)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_train_and_predict(self):
        train_data = ImageUtil.load_image_paths_and_labels(self.dataset_dir)
        self.assertEqual(len(train_data), 6)

        rf_model = RandomForest(descriptor_type=DescriptorType.SIFT)
        rf_model.train(train_data)
        self.assertTrue(rf_model.is_trained())

        test_image_path = os.path.join(self.class_a_path, "a_0.png")
        prediction = rf_model.predict(test_image_path)
        self.assertEqual(prediction, "class_a")

    def test_save_and_load_model(self):
        train_data = ImageUtil.load_image_paths_and_labels(self.dataset_dir)
        
        rf_original = RandomForest(descriptor_type=DescriptorType.SIFT)
        rf_original.train(train_data)
        self.assertTrue(rf_original.is_trained())
        rf_original.save_model(self.model_path)

        self.assertTrue(os.path.exists(self.model_path))

        rf_loaded = RandomForest.load_model(self.model_path)
        self.assertIsNotNone(rf_loaded)
        self.assertTrue(rf_loaded.is_trained())

        test_image_path = os.path.join(self.class_b_path, "b_0.png")
        prediction = rf_loaded.predict(test_image_path)
        self.assertEqual(prediction, "class_b")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    unittest.main()
