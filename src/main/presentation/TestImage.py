import os

import cv2
from matplotlib import pyplot as plt

from src.main.application.use_case.ImageUtil import ImageUtil
from src.main.domain.DescriptorType import DescriptorType


class TestImage:
    def _draw_keypoints(self, image, keypoints, title="Keypoints"):
        """Draws keypoints on an image and displays it."""
        img_with_keypoints = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()
        
    def run_visualization(self, base_dir, file_name, feature_dir, descriptor_type):
        """Extracts features from an image and visualizes the keypoints."""
        image_path = os.path.join(base_dir, file_name)
        print(f"Extracting {descriptor_type.name} features from {image_path}...")
        features = ImageUtil.extract_features(image_path, descriptor_type, use_gpu=True)

        if features:
            ImageUtil.save_features(feature_dir, file_name, features)
        keypoints_path = os.path.join(feature_dir, f"{file_name}_keypoints.npy")
        descriptors_path = os.path.join(feature_dir, f"{file_name}_descriptors.npy")
        loaded_features = ImageUtil.load_features(keypoints_path, descriptors_path, image_path, "label")

        if loaded_features:
            keypoints, descriptors = loaded_features.keypoints, loaded_features.descriptors

            print(f"Found {len(keypoints)} keypoints.")
            image = cv2.imread(image_path)
            self._draw_keypoints(image, keypoints, title=f"{descriptor_type.name} Keypoints (GPU)")
        else:
            print("No keypoints were found for the given image and descriptors.")


if __name__ == '__main__':
    base_dir = "/home/patrick/Documents/project/img_search/asset/dataset/clothes/train"
    file_name = "a824deb0-6985-4b11-a987-74d47f5fc33e.jpg"
    feature_dir = "/home/patrick/Documents/project/img_search/asset/dataset/clothes/feature"
    test_image_search = TestImage()
    descriptor_type = DescriptorType.SIFT_ADHOC
    test_image_search.run_visualization(base_dir, file_name, feature_dir, descriptor_type)