from enum import Enum
import cv2 as cv

class ClassifierType(Enum):
    SVM = 1
    LOGISTIC_REGRESSION = 2

    def create_classifier(self):
        if self == ClassifierType.SVM:
            return cv.ml.SVM_create()
        elif self == ClassifierType.LOGISTIC_REGRESSION:
            return cv.ml.LogisticRegression_create()
        else:
            raise ValueError("Unsupported classifier type")