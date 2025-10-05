import cv2 as cv
import numpy as np
from enum import Enum
from src.main.application.use_case.ImageUtil import ImageUtil


class FeatureAlgorithm(Enum):
    SIFT=1
    ORB=2
    KAZE=3
    AKAZE=4
    BRISK=5
    SIFT_FREAK=6

class ImageDescriptorAnalyzer:

    def __init__(self):
        pass


    @staticmethod
    def calculate_matching(keypoints1, descriptors1, keypoints2, descriptors2):
        if not all(x is not None and len(x) > 0 for x in [keypoints1, descriptors1, keypoints2, descriptors2]):
            return 0.0, 0.0

        brute_force_matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = brute_force_matcher.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    @staticmethod
    def calculate_match_to_keypoint_ratio(matches, keypoints1):
        return len(matches) / len(keypoints1) if len(keypoints1) > 0 else 0.0

    @staticmethod
    def calculate_average_match_distance(matches, keypoints1, keypoints2):
        distances = [np.linalg.norm(np.array(keypoints1[match.queryIdx].pt) - np.array(keypoints2[match.trainIdx].pt))
                     for match in matches]
        return np.average(distances) if distances else 0.0

    @staticmethod
    def extract_features(image_path, algorithm: FeatureAlgorithm, desired_keypoints: int = 5000):
        image = ImageUtil.load_grayscale_image(image_path)
        detector, descriptor = ImageDescriptorAnalyzer._create_feature_extractor(algorithm, desired_keypoints)

        if descriptor is not None:  # Composite detector like SIFT_FREAK
            keypoints = detector.detect(image, None)
            return descriptor.compute(image, keypoints)
        else:
            return detector.detectAndCompute(image, None)

    @staticmethod
    def _create_feature_extractor(algorithm: FeatureAlgorithm, desired_keypoints: int = 5000):
        if algorithm == FeatureAlgorithm.SIFT:
            return cv.SIFT_create(), None
        elif algorithm == FeatureAlgorithm.ORB:
            return cv.ORB_create(desired_keypoints), None
        elif algorithm == FeatureAlgorithm.KAZE:
            return cv.KAZE_create(), None
        elif algorithm == FeatureAlgorithm.AKAZE:
            return cv.AKAZE_create(), None
        elif algorithm == FeatureAlgorithm.BRISK:
            return cv.BRISK_create(), None
        elif algorithm == FeatureAlgorithm.SIFT_FREAK:
            detector = cv.SIFT_create()
            descriptor = cv.xfeatures2d.FREAK_create()
            return detector, descriptor
        
        raise ValueError(f"Unknown or unsupported feature algorithm: {algorithm}")
