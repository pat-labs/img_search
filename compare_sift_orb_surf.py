import os
import random
import time
from functools import wraps

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def sift_timing(img_path):
    # SIFT Timing
    sift = cv.SIFT_create(300)

    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"Could not open {img_path}")

    start_time = time.time()
    kp, des = sift.detectAndCompute(img, None)
    sift_time = time.time() - start_time
    sift_features = np.array(kp).shape[0]
    return [sift_time, sift_features]


def sift_test(img_path, compare_img_path, flag_function, reduce=True):
    # SIFT Brightness Matching or Rotation Matching
    calculate_dist = lambda c1, c2: ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
    if flag_function:
        calculate_dist = (
            lambda c1, c2: (
                (c1[0] - (c2[0] * (-1) + 720)) ** 2
                + (c1[1] - (c2[1] * (-1) + 480)) ** 2
            )
            ** 0.5
        )
    sift = cv.SIFT_create(1000000)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    p_matched = []
    dist_list = []

    img1 = cv.imread(img_path)
    img2 = cv.imread(compare_img_path)
    if img1 is None:
        raise ValueError(f"Could not open {img_path}")
    if img2 is None:
        raise ValueError(f"Could not open {compare_img_path}")

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    p_matched.append(len(matches) / len(kp1))

    if reduce:
        matches = matches[:500]
    distance = []
    for match in matches:
        c1 = kp1[match.queryIdx].pt
        c2 = kp2[match.trainIdx].pt
        dist = calculate_dist(c1, c2)
        distance.append(dist)
    dist_list.append(np.average(dist))

    return [p_matched, dist_list]


def orb_timing(img_path):
    # ORB Timing
    orb = cv.ORB_create(300)

    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"Could not open {img_path}")
    start_time = time.time()
    kp, des = orb.detectAndCompute(img, None)
    orb_time = time.time() - start_time
    orb_features = np.array(kp).shape[0]
    return [orb_time, orb_features]


def orb_test(img_path, compare_img_path, flag_function, reduce=True):
    # ORB Brightness Matching or Rotation Matching
    calculate_dist = lambda c1, c2: ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
    if flag_function:
        calculate_dist = (
            lambda c1, c2: (
                (c1[0] - (c2[0] * (-1) + 720)) ** 2
                + (c1[1] - (c2[1] * (-1) + 480)) ** 2
            )
            ** 0.5
        )
    orb = cv.ORB_create(1000000)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    p_matched = []
    dist_list = []

    img1 = cv.imread(img_path)
    img2 = cv.imread(compare_img_path)
    if img1 is None:
        raise ValueError(f"Could not open {img_path}")
    if img2 is None:
        raise ValueError(f"Could not open {compare_img_path}")
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    p_matched.append(len(matches) / len(kp1))

    if reduce:
        matches = matches[:500]
    distance = []
    for match in matches:
        c1 = kp1[match.queryIdx].pt
        c2 = kp2[match.trainIdx].pt
        dist = calculate_dist(c1, c2)
        distance.append(dist)
    dist_list.append(np.average(dist))

    return [p_matched, dist_list]


# TODO: separete funcionts and mix
def get_orb_sift_image_descriptors(search_img, idx_img):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # Initiate ORB detector
    orb = cv.ORB_create()
    # Find keypoints with ORB
    search_kp_orb = orb.detect(search_img, None)
    idx_kp_orb = orb.detect(idx_img, None)
    # Compute descriptors with SIFT
    search_kp_sift, search_des_sift = sift.compute(search_img, search_kp_orb)
    idx_kp_sift, idx_des_sift = sift.compute(idx_img, idx_kp_orb)
    return search_des_sift, idx_des_sift


def surf_timing(img_path):
    # SURF Timing
    surf = cv.xfeatures2d.SURF(2200)
    surf_time = 0
    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"Could not open {img_path}")

    start_time = time.time()
    kp, des = surf.detectAndCompute(img, None)
    surf_time = time.time() - start_time
    surf_features = np.array(kp).shape[0]
    return [surf_time, surf_features]


def surf_test(img_path, compare_img_path, flag_function, reduce=True):
    # SURF Brightness Matching or Rotation Matching
    calculate_dist = lambda c1, c2: ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
    if flag_function:
        calculate_dist = (
            lambda c1, c2: (
                (c1[0] - (c2[0] * (-1) + 720)) ** 2
                + (c1[1] - (c2[1] * (-1) + 480)) ** 2
            )
            ** 0.5
        )
    surf = cv.SURF_create()
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    p_matched = []
    dist_list = []

    img1 = cv.imread(img_path)
    img2 = cv.imread(compare_img_path)
    if img1 is None:
        raise ValueError(f"Could not open {img_path}")
    if img2 is None:
        raise ValueError(f"Could not open {compare_img_path}")
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    p_matched.append(len(matches) / len(kp1))

    if reduce:
        matches = matches[:500]
    distance = []
    for match in matches:
        c1 = kp1[match.queryIdx].pt
        c2 = kp2[match.trainIdx].pt
        dist = calculate_dist(c1, c2)
        distance.append(dist)
    dist_list.append(np.average(dist))

    return [p_matched, dist_list]


def make_plot(sift_result, surf_result, orb_result, title, label_x, label_y):
    fig = plt.figure()
    sns.set()
    ax = fig.add_axes([0, 0, 1, 1])
    methods = ["SIFT", "SURF", "ORB"]
    times = [sift_result, surf_result, orb_result]
    ax.barh(methods, times, color=("green", "blue", "orange"))
    ax.set_ylabel(label_y)
    ax.set_xlabel(label_x)
    ax.set_title(title)
    plt.show()


def main(img_folder, brightness_folder, rotation_folder):
    sift_results = list()
    surf_results = list()
    orb_results = list()

    for img_name in os.listdir(img_folder):
        sift_results.append(sift_timing(img_folder + img_name))
        surf_results.append(surf_timing(img_folder + img_name))
        orb_results.append(orb_timing(img_folder + img_name))

    sift_time = np.average([item[0] for item in sift_results])
    surf_time = np.average([item[0] for item in surf_results])
    orb_time = np.average([item[0] for item in orb_results])
    make_plot(
        sift_time * 1000,
        surf_time * 1000,
        orb_time * 1000,
        "Average time to compute ~300 Key-Point Descriptors",
        "Time (ms)",
        "Feature Extractor",
    )

    sift_features2 = np.average([item[1] for item in sift_results])
    surf_features2 = np.average([item[1] for item in surf_results])
    orb_features2 = np.average([item[1] for item in orb_results])
    make_plot(
        sift_features2,
        surf_features2,
        orb_features2,
        "Average total number of extracted key-points per image",
        "Number of key-points",
        "Feature Extractor",
    )

    sift_brightness = list()
    surf_brightness = list()
    orb_brightness = list()
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        bright_img_path = os.path.join(brightness_folder, img_name)

        sift_brightness = sift_test(img_path, bright_img_path, False)
        surf_brightness = surf_test(img_path, bright_img_path, False)
        orb_brightness = orb_test(img_path, bright_img_path, False)

    sift_per_b = np.average(item[1] for item in sift_brightness)
    sift_dist_b = np.average(item[2] for item in sift_brightness)
    surf_per_b = np.average(item[1] for item in surf_brightness)
    surf_dist_b = np.average(item[2] for item in surf_brightness)
    orb_per_b = np.average(item[1] for item in orb_brightness)
    orb_dist_b = np.average(item[2] for item in orb_brightness)
    make_plot(
        sift_per_b,
        surf_per_b,
        orb_per_b,
        "Average Percentage of Matched Keypoints for Brightened Image",
        "Percentage",
        "Feature Extractor",
    )
    make_plot(
        sift_dist_b,
        surf_dist_b,
        orb_dist_b,
        "Average Percentage of Matched Keypoints for Rotated Image",
        "Percentage",
        "Feature Extractor",
    )

    sift_rotation = list()
    surf_rotation = list()
    orb_rotation = list()
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        rotated_img_path = os.path.join(rotation_folder, img_name)

        sift_rotation = sift_test(img_path, rotated_img_path, True)
        surf_rotation = surf_test(img_path, rotated_img_path, True)
        orb_rotation = orb_test(img_path, rotated_img_path, True)

    sift_per_r = np.average(item[1] for item in sift_rotation)
    sift_dist_r = np.average(item[2] for item in sift_rotation)
    surf_per_r = np.average(item[1] for item in surf_rotation)
    surf_dist_r = np.average(item[2] for item in surf_rotation)
    orb_per_r = np.average(item[1] for item in orb_rotation)
    orb_dist_r = np.average(item[2] for item in orb_rotation)
    make_plot(
        sift_per_r,
        surf_per_r,
        orb_per_r,
        "Average Drift of Matched Keypoints for Brightened Image",
        "Pixels",
        "Feature Extractor",
    )
    make_plot(
        sift_dist_r,
        surf_dist_r,
        orb_dist_r,
        "Average Drift of Matched Keypoints for Rotated Image",
        "Pixels",
        "Feature Extractor",
    )


def test_functions():
    img_folder = "assets"
    img_name = "annapurna.jpg"
    img_path = os.path.join(img_folder, img_name)
    bright_img_path = os.path.join(img_folder, img_name)
    rotated_img_path = os.path.join(img_folder, img_name)

    brightness = random.randint(1, 359)
    increase_brightness(img_path, bright_img_path, brightness)

    angle = random.randint(1, 359)
    rotate_image(img_path, rotated_img_path, angle)

    sift_time, sift_features2 = sift_timing(img_path)
    surf_time, surf_features2 = surf_timing(img_path)
    orb_time, orb_features2 = orb_timing(img_path)
    make_plot(
        sift_time * 1000,
        surf_time * 1000,
        orb_time * 1000,
        "Average time to compute ~300 Key-Point Descriptors",
        "Time (ms)",
        "Feature Extractor",
    )

    sift_number_features = np.average(sift_features2)
    surf_number_features = np.average(surf_features2)
    orb_number_features = np.average(orb_features2)
    make_plot(
        sift_number_features,
        surf_number_features,
        orb_number_features,
        "Average total number of extracted key-points per image",
        "Number of key-points",
        "Feature Extractor",
    )


if __name__ == "__main__":
    img_folder = "dataset//flowers//train//sunflower"
    brightness_folder = "dataset//flowers//brightness"
    rotated_folder = "dataset//flowers//rotation"

    # setup(img_folder, brightness_folder, rotation_folder)
    # main(img_folder, brightness_folder, rotation_folder)
    test_functions()
