import logging

import cv2
import numpy as np
from matplotlib import pyplot as plt

import pysift

logger = logging.getLogger(__name__)


def explanatory_sift(img1, img2):
    MIN_MATCH_COUNT = 10

    # Compute SIFT keypoints and descriptors
    kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
    kp2, des2 = pysift.computeKeypointsAndDescriptors(img2)

    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        # Draw detected template in scene image
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        h1, w1 = img1.shape
        h2, w2 = img2.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif : hdif + h1, :w1, i] = img1
            newimg[:h2, w1 : w1 + w2, i] = img2

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        plt.imshow(newimg)
        plt.show()
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))


def sift_algorithm(img):
    if img is not None:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray_img, None)
        img = cv2.drawKeypoints(gray_img, kp, img)
        cv2.imwrite("result/sift_keypoints_1.jpg", img)
    else:
        print("Failed to read image. Please check the URL and internet connection.")

    img = cv2.drawKeypoints(
        gray_img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite("result/sift_keypoints_2.jpg", img)


def sift_algorithm_white_background(img):
    if img is not None:
        # Create a white background image with the same size as the original image
        white_bg = np.full_like(img, 255, dtype=np.uint8)  # 255 represents white color
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray_img, None)
        img = cv2.drawKeypoints(
            white_bg, kp, white_bg, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )
        cv2.imwrite("result/sift_keypoints_1.jpg", img)
    else:
        print("Failed to read image. Please check the URL and internet connection.")

    img = cv2.drawKeypoints(
        gray_img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite("result/sift_keypoints_2.jpg", img)


def sift_matches(img1, img2):
    if img1 is not None and img2 is not None:
        sift = cv2.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
        # feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        img3 = cv2.drawMatches(
            img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2
        )

        cv2.imwrite("result/sift_matches.jpg", img3)
    else:
        print("Failed to read image. Please check the URL and internet connection.")
