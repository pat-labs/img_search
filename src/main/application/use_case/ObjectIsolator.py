from dataclasses import dataclass
from typing import List, Optional

import cv2 as cv
import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


class ObjectIsolator:
    @staticmethod
    def get_bounding_boxes(img: np.ndarray) -> List[BoundingBox]:
        try:
            mask = np.zeros(img.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            rect_margin = 5
            rect = (rect_margin, rect_margin, img.shape[1] - rect_margin * 2, img.shape[0] - rect_margin * 2)

            cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            contours, _ = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if not contours:
                print("No contours found after GrabCut.")
                return []

            bounding_boxes = []
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                bounding_boxes.append(BoundingBox(x=x, y=y, width=w, height=h))

            return bounding_boxes

        except Exception as e:
            print(f"An error occurred during object isolation: {e}")
            return []

    @staticmethod
    def get_isolate_object(img: np.ndarray, bounding_boxes: List[BoundingBox]) -> List[np.ndarray]:
        isolated_objects = []
        for box in bounding_boxes:
            cropped_img = img[box.y: box.y + box.height, box.x: box.x + box.width]
            isolated_objects.append(cropped_img)
        return isolated_objects

    @staticmethod
    def find_object_by_template(scene_image: np.ndarray, ref_image: np.ndarray, min_match_count: int = 10) -> Optional[BoundingBox]:
        """
        Finds a reference object in a scene image using SIFT feature matching and returns its bounding box.
        :param scene_image: The image to search within.
        :param ref_image: A template image of the object to find.
        :param min_match_count: The minimum number of good matches required to consider the object found.
        :return: A BoundingBox if the object is found, otherwise None.
        """
        try:
            # 1. Initialize SIFT detector
            sift = cv.SIFT_create()

            # 2. Find keypoints and descriptors in both images
            kp1, des1 = sift.detectAndCompute(ref_image, None)
            kp2, des2 = sift.detectAndCompute(scene_image, None)

            if des1 is None or des2 is None:
                return None

            # 3. Match features using a FLANN based matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # 4. Filter good matches using Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # 5. If enough good matches are found, find the object
            if len(good_matches) > min_match_count:
                # Get coordinates of matched keypoints
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find the homography matrix
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                if M is None: return None

                # Project the corners of the reference image to find the bounding box in the scene
                h, w = ref_image.shape[:2]
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, M)
                
                # Get an axis-aligned bounding box from the projected corners
                x, y, w, h = cv.boundingRect(dst)
                return BoundingBox(x=x, y=y, width=w, height=h)

            return None
        except Exception as e:
            print(f"An error occurred during template matching: {e}")
            return None
