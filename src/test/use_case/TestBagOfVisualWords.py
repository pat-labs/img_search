import unittest
import os
import tempfile
import shutil
import numpy as np
import cv2

from src.main.application.use_case.BagOfVisualWords import BagOfVisualWords, KNodes, DescriptorType, ClassifierType
from src.main.application.use_case.ImageDescriptorAnalyzer import ImageDescriptorAnalyzer
from src.main.application.use_case.ImageUtil import ImageUtil

class TestBagOfVisualWords(unittest.TestCase):

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
            cv2.rectangle(img_a, (10, 10 + i), (25, 25 + i), 255, -1)  # A filled rectangle, slightly different each time
            
            # Image B: White circle on black background
            img_b = np.zeros((50, 50), dtype=np.uint8)
            cv2.circle(img_b, (25, 25), 10 + i, 255, -1) # A filled circle, slightly different each time
            cv2.imwrite(os.path.join(self.class_a_path, f"a_{i}.png"), img_a)
            cv2.imwrite(os.path.join(self.class_b_path, f"b_{i}.png"), img_b)

        self.descriptor_type = DescriptorType.SIFT
        self.classifier_type = ClassifierType.SVM
        path_and_labels = ImageUtil.load_image_paths_and_labels(self.dataset_dir)
        self.all_descriptors = ImageDescriptorAnalyzer.extract_features_serial([item.path for item in path_and_labels], self.descriptor_type)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_train_and_predict_with_save(self):
        train_data = ImageUtil.load_image_paths_and_labels(self.dataset_dir)
        self.assertEqual(len(train_data), 6)

        bovw = BagOfVisualWords(
            k=KNodes.K16,
            classifier_type=self.classifier_type,
            descriptor_type=self.descriptor_type
        )

        bovw.train(train_data, self.all_descriptors)
        self.assertTrue(bovw.is_trained())
        bovw.save_model(self.model_dir)

        # Test if it can predict one of its training images correctly
        test_image_path = os.path.join(self.class_a_path, "a_0.png")
        prediction = bovw.predict(test_image_path)
        self.assertEqual(prediction, "class_a")

        # Find the saved model files
        saved_files = os.listdir(self.model_dir)
        vocab_path = os.path.join(self.model_dir, next(f for f in saved_files if f.endswith(".npz")))
        classifier_path = os.path.join(self.model_dir, next(f for f in saved_files if f.endswith(".xml")))

        # 2. Load the model into a new instance
        bovw_loaded = BagOfVisualWords.load_model(vocab_path, classifier_path)
        self.assertIsNotNone(bovw_loaded)
        self.assertTrue(bovw_loaded.is_trained())

        # 3. Verify the loaded model can predict correctly
        test_image_path = os.path.join(self.class_b_path, "b_0.png")
        prediction = bovw_loaded.predict(test_image_path)
        self.assertEqual(prediction, "class_b")

if __name__ == '__main__':
    unittest.main()
