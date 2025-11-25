import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from src.main.application.use_case.AnisotropicSIFT import anisotropicSiftComputeKeypointsAndDescriptors
from src.main.application.use_case.ImageUtil import ImageUtil


class TestAnisotropicSIFT(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory and create a meaningful test image."""
        self.test_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.test_dir, "test_image.png")

        # Create an image with a distinct shape (a white star) to ensure features can be detected.
        img = np.zeros((256, 256), dtype=np.uint8)
        pts = np.array([[128, 50], [158, 110], [228, 110], [178, 150], [198, 210], [128, 170], [58, 210], [78, 150], [28, 110], [98, 110]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], 255)
        cv2.imwrite(self.image_path, img)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_sift(self):
        """Test that keypoints and descriptors are generated correctly."""
        img = ImageUtil.load_grayscale_image(self.image_path)
        keypoints, descriptors = anisotropicSiftComputeKeypointsAndDescriptors(img)

        # 1. Assert that keypoints and descriptors were found
        self.assertIsNotNone(keypoints)
        self.assertIsNotNone(descriptors)
        self.assertGreater(len(keypoints), 0, "Should find at least one keypoint in the test image.")

        # 2. Assert that the number of keypoints matches the number of descriptors
        self.assertEqual(len(keypoints), len(descriptors))

        # 3. Assert that the descriptors have the correct dimension (128 for SIFT)
        self.assertEqual(descriptors.shape[1], 128)

if __name__ == '__main__':
    unittest.main()