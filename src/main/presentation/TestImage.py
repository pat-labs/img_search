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
        
    def run_visualization(self, image_path, descriptor_type):
        """Extracts features from an image and visualizes the keypoints."""
        print(f"Extracting {descriptor_type.name} features from {image_path}...")
        features = ImageUtil.extract_features(image_path, descriptor_type)
        keypoints, descriptors = features.keypoints, features.descriptors

        if keypoints:
            print(f"Found {len(keypoints)} keypoints.")
            image = cv2.imread(image_path)
            self._draw_keypoints(image, keypoints, title=f"{descriptor_type.name} Keypoints")
        else:
            print("No keypoints were found for the given image and descriptors.")


if __name__ == '__main__':
    image_path = "/home/patrick/Documents/project/img_search/asset/dataset/clothes/train/a824deb0-6985-4b11-a987-74d47f5fc33e.jpg"
    test_image_search = TestImage()
    descriptor_type = DescriptorType.SIFT_ADHOC
    test_image_search.run_visualization(image_path, descriptor_type)