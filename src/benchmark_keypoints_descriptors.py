import csv
import time

import numpy as np
import pandas as pd

from algorithms_keypoints_descriptors import (brightnessMatch, drawKeypoints,
                                              execAkaze, execBrisk, execKaze,
                                              execOrb, execSift, execSiftFreak,
                                              execStartBrief,
                                              increaseBrightness, readImage,
                                              rotatedMatch, rotateImage)
from plot import barPlot

labels = ["SIFT", "ORB", "KAZE", "AKAZE", "BRISK", "SIFT_FREAK", "START_BRIEF"]
algorithms_perfomance_path = "result/algorithms_performance.csv"
match_bright_path = "result/match_bright.csv"
match_rotate_path = "result/match_rotate.csv"


def timeIt(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return result, elapsed_time


def process():
    """
    1. Create variations in images
    2. Benchmark
    3. Plot results
    """

    img_path = "asset/test"
    destiny_path = "result/test"
    img_extension = ".jpg"

    functions = [
        execSift,
        execOrb,
        execKaze,
        execAkaze,
        execBrisk,
        execSiftFreak,
        execStartBrief,
    ]

    # 1-254
    brightness = 150
    increaseBrightness(
        img_path + img_extension, img_path + "_bright" + img_extension, brightness
    )
    # 1-359
    angle = 180
    rotateImage(img_path + img_extension, img_path + "_rotate" + img_extension, angle)

    header_1 = ["INDEX", "LABEL", "FEATURES", "TIME_LAPSE"]
    data_1 = []
    header_2 = ["INDEX", "MATCH_KEYPOINTS", "MATCH_DESCRIPTORS", "TIME_LAPSE"]
    data_2 = []
    data_3 = []
    i = 0

    for func in functions:
        image = readImage(img_path + img_extension, 80, 80)
        keypoints_descriptors_normal, time_lapse_normal = timeIt(func, image)
        drawKeypoints(
            img_path + img_extension,
            destiny_path + "_" + labels[i] + img_extension,
            keypoints_descriptors_normal[0],
        )

        image_bright = readImage(img_path + "_bright" + img_extension)
        keypoints_descriptors_bright, time_lapse_bright = timeIt(func, image_bright)
        drawKeypoints(
            img_path + "_bright" + img_extension,
            destiny_path + "_bright_" + labels[i] + img_extension,
            keypoints_descriptors_bright[0],
        )

        image_rotate = readImage(img_path + "_rotate" + img_extension)
        keypoints_descriptors_rotate, time_lapse_rotate = timeIt(func, image_rotate)
        drawKeypoints(
            img_path + "_rotate" + img_extension,
            destiny_path + "_rotate_" + labels[i] + img_extension,
            keypoints_descriptors_rotate[0],
        )

        features_normal = np.array(keypoints_descriptors_normal[0]).shape[0]
        features_bright = np.array(keypoints_descriptors_bright[0]).shape[0]
        features_rotate = np.array(keypoints_descriptors_rotate[0]).shape[0]

        data_1.append([labels[i], "NORMAL", features_normal, time_lapse_normal])
        data_1.append([labels[i], "BRIGHT", features_bright, time_lapse_bright])
        data_1.append([labels[i], "ROTATE", features_rotate, time_lapse_rotate])

        keypoints_descriptors_bright_match, time_lapse_bright_match = timeIt(
            brightnessMatch,
            keypoints_descriptors_normal[0],
            keypoints_descriptors_normal[1],
            keypoints_descriptors_bright[0],
            keypoints_descriptors_bright[1],
        )
        keypoints_descriptors_rotateMatch, time_lapse_rotateMatch = timeIt(
            rotatedMatch,
            keypoints_descriptors_normal[0],
            keypoints_descriptors_normal[1],
            keypoints_descriptors_rotate[0],
            keypoints_descriptors_rotate[1],
        )

        data_2.append(
            [
                labels[i],
                keypoints_descriptors_bright_match[0],
                keypoints_descriptors_bright_match[1],
                time_lapse_bright_match,
            ]
        )
        data_3.append(
            [
                labels[i],
                keypoints_descriptors_rotateMatch[0],
                keypoints_descriptors_rotateMatch[1],
                time_lapse_rotateMatch,
            ]
        )

        i += 1

    df_1 = pd.DataFrame(data_1, columns=header_1)
    df_1.to_csv(algorithms_perfomance_path)

    df_2 = pd.DataFrame(data_2, columns=header_2)
    df_2.to_csv(match_bright_path)

    df_3 = pd.DataFrame(data_3, columns=header_2)
    df_3.to_csv(match_rotate_path)


def plotPerformanceAlgoprithm(data_path):
    with open(data_path, newline="") as csvfile:
        data_1 = list(csv.reader(csvfile))
    data_1.pop(0)

    i = 0
    for idx in range(0, len(data_1), 3):
        data = [data_1[idx], data_1[idx + 1], data_1[idx + 2]]

        legend_data = [str(item[3]) for item in data]

        barPlot(
            [float(item[4]) for item in data],
            [str(item[2]) for item in data],
            "Performance algorithm: " + labels[i],
            "Type (image)",
            "Time (ms)",
            ["Features"],
            [legend_data],
            "result/" + labels[i] + ".jpg",
        )
        i += 1


def plotMatch(data_path, title):
    with open(data_path, newline="") as csvfile:
        data = list(csv.reader(csvfile))
    data.pop(0)

    legend_data = [[item[2][:5], item[3][:5]] for item in data]
    t_legend_data = list(map(list, zip(*legend_data)))

    barPlot(
        [float(item[4]) for item in data],
        [str(item[1]) for item in data],
        "Match: " + title,
        "Algorithm",
        "Time (ms)",
        ["keypoints", "descriptors"],
        t_legend_data,
        "result/match_" + title + ".jpg",
    )


if __name__ == "__main__":
    process()
    plotPerformanceAlgoprithm(algorithm_performance_path)
    plotMatch(match_bright_path, "bright")
    plotMatch(match_rotate_path, "rotate")
