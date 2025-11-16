from enum import Enum


class DescriptorType(Enum):
    SIFT = 1
    ORB = 2
    KAZE = 3
    AKAZE = 4
    BRISK = 5
    ANSIOTROPIC_SIFT = 6