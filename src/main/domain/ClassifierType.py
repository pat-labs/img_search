from enum import Enum

import cv2 as cv


class ClassifierType(Enum):
    SVM = 1
    LOGISTIC_REGRESSION = 2