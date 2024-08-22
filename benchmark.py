import os

import cv2 as cv
import numpy as np

from stich.PanoramaStitching import DetectAndComputeAlgorithm
from utils import (
    get_data_from_csv,
    read_image,
    save_descriptors,
    save_keypoints,
    save_to_csv,
    set_plot,
    set_plot_multiple_values,
    time_it,
)

RESULT_PATH = "result/image"


def detect_keypoints_with_star(img, desired_keypoints=100):
    # Initial parameters
    max_size = 45
    response_threshold = 30
    line_threshold_projected = 10
    line_threshold_binarized = 8
    suppress_nonmax_size = 5

    while True:
        star = cv.xfeatures2d.StarDetector_create(
            maxSize=max_size,
            responseThreshold=response_threshold,
            lineThresholdProjected=line_threshold_projected,
            lineThresholdBinarized=line_threshold_binarized,
            suppressNonmaxSize=suppress_nonmax_size,
        )
        kp = star.detect(img, None)

        if len(kp) >= desired_keypoints:
            break

        # Adjust parameters based on your desired goal
        response_threshold -= 5  # Example: decrease responseThreshold

    return kp


def exec_sift(image_path, features=1000):
    img, gray = read_image(image_path)

    sift = cv.SIFT_create(features)
    kp, des = sift.detectAndCompute(gray, None)

    img = cv.drawKeypoints(
        gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    file_name = os.path.basename(image_path)
    result_path = os.path.join(RESULT_PATH, file_name + "_SIFT.jpg")
    cv.imwrite(result_path, img)

    return (kp, des)


def exec_surf(image_path):
    img, gray = read_image(image_path)

    surf = cv.xfeatures2d_SURF.create()
    kp, des = surf.detectAndCompute(gray, None)

    img = cv.drawKeypoints(
        gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    file_name = os.path.basename(image_path)
    result_path = os.path.join(RESULT_PATH, file_name + "_SURF.jpg")
    cv.imwrite(result_path, img)

    return (kp, des)


def exec_orb(image_path, features=1000):
    img, gray = read_image(image_path)

    orb = cv.ORB_create(features)
    kp, des = orb.detectAndCompute(img, None)

    img = cv.drawKeypoints(
        gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    file_name = os.path.basename(image_path)
    result_path = os.path.join(RESULT_PATH, file_name + "_ORB.jpg")
    cv.imwrite(result_path, img)

    return (kp, des)


def exec_kaze(image_path):
    img, gray = read_image(image_path)

    kaze = cv.KAZE.create()
    kp, des = kaze.detectAndCompute(gray, None)

    img = cv.drawKeypoints(
        gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    file_name = os.path.basename(image_path)
    result_path = os.path.join(RESULT_PATH, file_name + "_KAZE.jpg")
    cv.imwrite(result_path, img)

    return (kp, des)


def exec_akaze(image_path):
    img, gray = read_image(image_path)

    akaze = cv.AKAZE_create()
    kp, des = akaze.detectAndCompute(gray, None)

    img = cv.drawKeypoints(
        gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    file_name = os.path.basename(image_path)
    result_path = os.path.join(RESULT_PATH, file_name + "_AKAZE.jpg")
    cv.imwrite(result_path, img)

    return (kp, des)


def exec_brisk(image_path):
    img, gray = read_image(image_path)

    brisk = cv.BRISK_create()
    kp, des = brisk.detectAndCompute(gray, None)

    img = cv.drawKeypoints(
        gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    file_name = os.path.basename(image_path)
    result_path = os.path.join(RESULT_PATH, file_name + "_BRISK.jpg")
    cv.imwrite(result_path, img)

    return (kp, des)


def exec_brief_orb(image_path):
    img, gray = read_image(image_path)

    orb = cv.ORB_create()
    kp = orb.detect(gray, None)
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp, des = brief.compute(img, kp)

    img = cv.drawKeypoints(
        gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    file_name = os.path.basename(image_path)
    result_path = os.path.join(RESULT_PATH, file_name + "_BRIEF.jpg")
    cv.imwrite(result_path, img)

    return (kp, des)


def exec_freak_sift(image_path):
    img, gray = read_image(image_path)

    sift = cv.SIFT_create()
    kp = sift.detect(gray, None)
    freak = cv.xfeatures2d.FREAK_create()
    kp, des = freak.compute(img, kp)

    img = cv.drawKeypoints(
        gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    file_name = os.path.basename(image_path)
    result_path = os.path.join(RESULT_PATH, file_name + "_FREAK.jpg")
    cv.imwrite(result_path, img)

    return (kp, des)


def rotated_match(kp1, des1, kp2, des2):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    p_matched = []
    dist_list = []
    distance = []

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    p_matched.append(len(matches) / len(kp1))

    for match in matches:
        c1 = kp1[match.queryIdx].pt
        c2 = kp2[match.trainIdx].pt
        dist = (
            (c1[0] - (c2[0] * (-1) + 720)) ** 2 + (c1[1] - (c2[1] * (-1) + 480)) ** 2
        ) ** 0.5
        distance.append(dist)
    dist_list.append(np.average(dist))
    per_r = np.average(p_matched)
    dist_r = np.average(dist_list)

    return per_r, dist_r


def brightness_match(kp1, des1, kp2, des2):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    p_matched = []
    dist_list = []
    distance = []

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    p_matched.append(len(matches) / len(kp1))

    for match in matches:
        c1 = kp1[match.queryIdx].pt
        c2 = kp2[match.trainIdx].pt
        dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
        distance.append(dist)
    dist_list.append(np.average(dist))
    per_r = np.average(p_matched)
    dist_r = np.average(dist_list)

    return per_r, dist_r


def test_functions(result_path, transform_result_path):
    image_path = "assets//annapurna_1.jpg"
    brightness_image_path = "assets//annapurna_1_brightness.jpg"
    rotated_image_path = "assets//annapurna_1_rotated.jpg"
    labels = ["SIFT", "ORB", "KAZE", "AKAZE", "BRISK", "BRIEF_ORB", "FREAK_SIFT"]
    functions = [
        exec_sift,
        exec_orb,
        exec_kaze,
        exec_akaze,
        exec_brisk,
        exec_brief_orb,
        exec_freak_sift,
    ]
    header_1 = ["TYPE", "FEATURES_ALGORITHM", "FEATURES", "TIME_COST"]
    header_2 = [
        "TYPE",
        "FEATURES_ALGORITHM",
        "AVERAGE PERCENTAGE OF MATCHED KEYPOINTS",
        "AVERAGE DRIFT OF MATCHED KEYPOINTS",
        "TIME_COST",
    ]

    data_1 = list()
    data_2 = list()
    i = 0
    for func in functions:
        res_1, time_lapse_1 = time_it(func, image_path)
        res_2, time_lapse_2 = time_it(func, brightness_image_path)
        res_3, time_lapse_3 = time_it(func, rotated_image_path)
        features_1 = np.array(res_1[0]).shape[0]
        features_2 = np.array(res_2[0]).shape[0]
        features_3 = np.array(res_3[0]).shape[0]
        res_4, time_lapse_4 = time_it(
            brightness_match, res_1[0], res_1[1], res_2[0], res_2[1]
        )
        res_5, time_lapse_5 = time_it(
            rotated_match, res_1[0], res_1[1], res_3[0], res_3[1]
        )
        data_1.append(["NORMAL", labels[i], features_1, time_lapse_1])
        data_1.append(["INCREASE_BRIGHTNESS", labels[i], features_2, time_lapse_2])
        data_1.append(["ROTATED", labels[i], features_3, time_lapse_3])

        data_2.append(
            ["NORMAL_VS_BRIGHTNESS", labels[i], res_4[0], res_4[1], time_lapse_4]
        )
        data_2.append(
            ["NORMAL_VS_ROTATED", labels[i], res_5[0], res_5[1], time_lapse_5]
        )
        i += 1

    save_to_csv(header_1, data_1, result_path)
    save_to_csv(header_2, data_2, transform_result_path)


def generate_features_file():
    folder_path = "dataset/flowers/train/sunflower"
    result_path = "dataset/flowers/features/sunflower"
    labels = ["SIFT", "ORB", "KAZE", "AKAZE", "BRISK", "SIFT_BRIEF", "FREAK_SIFT"]
    functions = [
        exec_sift,
        exec_orb,
        exec_kaze,
        exec_akaze,
        exec_brisk,
        exec_brief_orb,
        exec_freak_sift,
    ]

    i = 0
    for func in functions:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    image_path = os.path.join(root, file)
                    print("File: " + image_path)
                    res_1 = func(image_path)
                    keypoints_path = os.path.join(
                        result_path, file + "_" + labels[i] + "_KEYPOINTS.pkl"
                    )
                    descriptors_path = os.path.join(
                        result_path, file + "_" + labels[i] + "_DESCRIPTORS.pkl"
                    )
                    save_keypoints(keypoints_path, res_1[0])
                    save_descriptors(descriptors_path, res_1[1])
        i += 1


def plot_results(result_path, transform_result_path):
    labels = ["SIFT", "ORB", "KAZE", "AKAZE", "BRISK", "SIFT_BRIEF", "FREAK_SIFT"]
    header_1, data_1 = get_data_from_csv(result_path)
    algorithm_time = list()
    i = 0
    features_labels = [
        "SIFT",
        "ORB",
        "KAZE",
        "AKAZE",
        "BRISK",
        "SIFT_BRIEF",
        "FREAK_SIFT",
    ]
    for idx in range(0, len(data_1), 3):
        algorithm_time.append((data_1[idx][3], data_1[idx + 1][3], data_1[idx + 2][3]))
        features_labels[i] = (
            f"{features_labels[i]}\n({data_1[idx][2]}, {data_1[idx + 1][2]}, {data_1[idx + 2][2]})"
        )
        i += 1

    set_plot_multiple_values(
        algorithm_time,
        ["NORMAL", "INCREASE_BRIGHTNESS", "ROTATED"],
        features_labels,
        "Average time algorithms",
        "Features",
        "Time (ms)",
        "result/plot/compare_average_algorithms_time.png",
    )
    header_2, data_2 = get_data_from_csv(transform_result_path)
    compare_matches_time = list()
    for idx in range(0, len(data_2), 2):
        compare_matches_time.append((data_2[idx][4], data_2[idx + 1][4]))
    set_plot_multiple_values(
        compare_matches_time,
        ["NORMAL_VS_BRIGHTNESS", "NORMAL_VS_ROTATED"],
        labels,
        "Average time algorithms matches",
        "Features",
        "Time (ms)",
        "result/plot/compare_average_algorithms_matches_time.png",
    )

    compare_matches_percentange = list()
    compare_matches_drift = list()
    for idx in range(0, len(data_2), 2):
        compare_matches_percentange.append((data_2[idx][2], data_2[idx + 1][2]))
        compare_matches_drift.append((data_2[idx][3], data_2[idx + 1][3]))

    set_plot_multiple_values(
        compare_matches_percentange,
        ["NORMAL_VS_BRIGHTNESS", "NORMAL_VS_ROTATED"],
        labels,
        "Average Percentage of Matched Keypoints for Increase Brigthness Image",
        "Features",
        "Matches",
        "result/plot/compare_average_algorithms_matches_percentage.png",
    )
    set_plot_multiple_values(
        compare_matches_drift,
        ["NORMAL_VS_BRIGHTNESS", "NORMAL_VS_ROTATED"],
        labels,
        "Average Drift of Matched Keypoints for Rotated Image",
        "Features",
        "Matches",
        "result/plot/compare_average_algorithms_matches_drift.png",
    )


if __name__ == "__main__":
    result_path = "result/data/algorithm_comparison.csv"
    transform_result_path = "result/data/transform_comparison.csv"
    # test_functions(result_path, transform_result_path)
    plot_results(result_path, transform_result_path)
    # generate_features_file()
    # k, d = exec_brief_orb("assets//train//15238348741_c2fb12ecf2_m.jpg")
    # print(type(k))
    # print(k)
    # # print(type(k[0]))
    # # print(k[0])
    # print("====================================")
    # print(type(d))
    # print(d)
    # print(type(d[0]))
    # print(d[0])
