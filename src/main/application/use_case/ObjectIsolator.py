import cv2
import numpy as np
from dataclasses import dataclass

from src.main.application.use_case.ImageUtil import ImageUtil


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

class ObjectIsolator:
    @staticmethod
    def isolate_object(image_path: str) -> BoundingBox | None:
        try:
            img = ImageUtil.load_image_paths_and_labels(image_path)
            if img is None:
                print(f"Error: Could not load image from {image_path}")
                return None

            mask = np.zeros(img.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            rect_margin = 10
            rect = (rect_margin, rect_margin, img.shape[1] - rect_margin*2, img.shape[0] - rect_margin*2)

            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                print("No contours found after GrabCut.")
                return None

            largest_contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest_contour)

            return BoundingBox(x=x, y=y, width=w, height=h)

        except Exception as e:
            print(f"An error occurred during object isolation: {e}")
            return None
