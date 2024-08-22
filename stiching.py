import os

import cv2 as cv

from stich import HarrisCornerDetection
from stich.PanoramaStitching import (
    DetectAndComputeAlgorithm,
    MatcherAlgorithm,
    stitch_two_images,
)
from utils import save_to_csv, set_plot, time_it

harris_threshold = 0.20
ransacIterations = 100
ransacThreshold = 5


def base_stich(img_1_path, img_2_path, img_name, detector_algorithm, matcher_algorithm):
    img1 = cv.imread(img_1_path)
    img2 = cv.imread(img_2_path)

    sift_jp1_jp2_stitched, maxInliers, len_matches = stitch_two_images(
        img1,
        img2,
        siftDisplayName="",
        siftFilePath="",
        saveSiftMatches=False,
        ransacDisplayName="SIFT STICH",
        saveRansacMatches=True,
        ransacFilePath="result/image/ransac_" + img_name + ".jpg",
        ransacIterations=ransacIterations,
        ransacThreshold=ransacThreshold,
        detector_algorithm=detector_algorithm,
        matcher_algorithm=matcher_algorithm,
    )
    res_path = "result/image/stich_" + img_name + ".jpg"
    cv.imwrite(res_path, sift_jp1_jp2_stitched)
    # cv.imshow("Stitched Japan1-3", sift_jp1_jp2_stitched)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return res_path, maxInliers, len_matches


if __name__ == "__main__":
    labels = ["SIFT", "ORB", "KAZE", "AKAZE", "BRISK", "SIFT-BRIEF", "FREAK"]
    folder_path = ["assets/Japan1.jpg", "assets/Japan2.jpg", "assets/Japan3.jpg"]

    data = list()
    i = 0
    matcher_algorithm = MatcherAlgorithm.BF
    for detector in DetectAndComputeAlgorithm:
        res_1, time_lapse_1 = time_it(
            base_stich,
            folder_path[0],
            folder_path[1],
            detector.name + "_BF_" + str(i),
            detector,
            MatcherAlgorithm.BF,
        )
        data.append(
            [
                1,
                res_1[0],
                detector.name,
                matcher_algorithm.name,
                res_1[1],
                res_1[2],
                time_lapse_1,
            ]
        )
        i += 1

        res_2, time_lapse_2 = time_it(
            base_stich,
            res_1[0],
            folder_path[2],
            detector.name + "_BF_" + str(i),
            detector,
            MatcherAlgorithm.BF,
        )
        data.append(
            [
                2,
                res_2[0],
                detector.name,
                matcher_algorithm.name,
                res_2[1],
                res_2[2],
                time_lapse_2,
            ]
        )
        i += 1

    matcher_algorithm = MatcherAlgorithm.FLANN
    for detector in DetectAndComputeAlgorithm:
        res_1, time_lapse_1 = time_it(
            base_stich,
            folder_path[0],
            folder_path[1],
            detector.name + "_FLANN_" + str(i),
            detector,
            MatcherAlgorithm.FLANN,
        )
        data.append(
            [
                1,
                res_1[0],
                detector.name,
                matcher_algorithm.name,
                res_1[1],
                res_1[2],
                time_lapse_1,
            ]
        )
        i += 1

        res_2, time_lapse_2 = time_it(
            base_stich,
            res_1[0],
            folder_path[2],
            detector.name + "_FLANN_" + str(i),
            detector,
            MatcherAlgorithm.FLANN,
        )
        data.append(
            [
                2,
                res_2[0],
                detector.name,
                matcher_algorithm.name,
                res_2[1],
                res_2[2],
                time_lapse_2,
            ]
        )
        i += 1

    header = [
        "ROUND",
        "IMAGE_PATH",
        "FEATURES_ALGORITHM",
        "MATCHER_ALGORITHM",
        "MAX_INLIERS",
        "LEN_MATCHES",
        "TIME_COST",
    ]
    save_to_csv(header, data, "result/data/stiching_comparison.csv")


def plot_results(results_path):
    labels = ["SIFT", "ORB", "KAZE", "AKAZE", "BRISK", "SIFT_BRIEF", "FREAK"]
    header, data = get_data_from_csv(results_path)
    set_plot(
        [item[6] for item in data if item[0] == 1],
        labels,
        "Average time first round",
        "Features",
        "Time (ms)",
        "result/plot/average_comparison_firts_round.png",
    )
