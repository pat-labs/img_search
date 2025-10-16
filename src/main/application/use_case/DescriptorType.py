from enum import Enum
import cv2 as cv
import numpy as np


class DescriptorType(Enum):
    SIFT = 1
    ORB = 2
    KAZE = 3
    AKAZE = 4
    BRISK = 5

    def create_descriptor(self, desired_keypoints: int = 5000) -> tuple:
        if self == DescriptorType.SIFT:
            return cv.SIFT_create(desired_keypoints)
        elif self == DescriptorType.ORB:
            return cv.ORB_create(desired_keypoints)
        elif self == DescriptorType.KAZE:
            return cv.KAZE_create()
        elif self == DescriptorType.AKAZE:
            return cv.AKAZE_create()
        elif self == DescriptorType.BRISK:
            return cv.BRISK_create()
        raise ValueError(f"Unsupported feature algorithm: {self.name}")

    def create_matcher(self):
        if self in [DescriptorType.SIFT, DescriptorType.KAZE, DescriptorType.AKAZE, DescriptorType.BRISK]:
            return cv.BFMatcher(cv.NORM_L2)
        elif self == DescriptorType.ORB:
            return cv.BFMatcher(cv.NORM_HAMMING)
        raise ValueError(f"Unsupported matcher for algorithm: {self.name}")

    def get_descriptor_dtype(self):
        if self == [DescriptorType.SIFT, DescriptorType.KAZE]:
            return np.float32
        elif self == [DescriptorType.ORB, DescriptorType.AKAZE, DescriptorType.BRISK]:
            return np.uint8
        raise ValueError(f"Unsupported feature algorithm: {self.name}")