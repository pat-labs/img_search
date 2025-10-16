import unittest
import os
import tempfile
import shutil
import numpy as np
import cv2

from src.main.application.use_case.FileHandler import FileHandler
from src.main.application.use_case.ImageUtil import ImageUtil

class TestImageUtil(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        self.image_path = os.path.join(self.test_dir, "source.png")
        dummy_image = np.zeros((10, 10), dtype=np.uint8)
        cv2.imwrite(self.image_path, dummy_image)

        self.dataset_path = os.path.join(self.test_dir, "dataset")
        os.makedirs(os.path.join(self.dataset_path, "daisy"))
        os.makedirs(os.path.join(self.dataset_path, "rose"))
        cv2.imwrite(os.path.join(self.dataset_path, "daisy", "d1.png"), dummy_image)
        cv2.imwrite(os.path.join(self.dataset_path, "rose", "r1.png"), dummy_image)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_create_image_variances(self):
        result_path = os.path.join(self.test_dir, "transform_result")

        ImageUtil.create_image_variances(self.image_path, result_path)

        self.assertTrue(os.path.isdir(result_path))
        image_name, _ = os.path.splitext(os.path.basename(self.image_path))
        generated_files = FileHandler.find_files_by_name(result_path, image_name)
        generated_files_name = [os.path.basename(file) for file in generated_files]

        self.assertEqual(len(generated_files_name), 8)

        expected_suffixes = [
            "_rotation_15.png",
            "_rotation_45.png",
            "_rotation_90.png",
            "_brightness_negative_25.png",
            "_brightness_25.png",
            "_brightness_50.png",
            "_flip_horizontal.png",
            "_flip_vertical.png"
        ]
        base_filename = "source"
        for suffix in expected_suffixes:
            self.assertIn(base_filename + suffix, generated_files_name)

    def test_load_image_paths_and_labels(self):
        paths_and_labels = ImageUtil.load_image_paths_and_labels(self.dataset_path)

        self.assertEqual(len(paths_and_labels), 2)

        labels = sorted([item.label for item in paths_and_labels])
        self.assertEqual(labels, ["daisy", "rose"])

        path_ends = sorted([os.path.basename(item.path) for item in paths_and_labels])
        self.assertEqual(path_ends, ["d1.png", "r1.png"])

    def test_load_grayscale_image(self):
        image = ImageUtil.load_grayscale_image(self.image_path)
        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (10, 10))

        with self.assertRaises(ValueError):
            ImageUtil.load_grayscale_image("non_existent_file.png")

if __name__ == '__main__':
    unittest.main()
