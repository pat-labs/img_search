import os
from typing import List, Tuple, Union

import cv2 as cv
import numpy as np
from pydantic import BaseModel
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from algorithms_keypoints_descriptors import execSift, readImage

train_data_path = "asset/flower/train"
test_data_path = "asset/flower/test"

# Create directories for training and testing datasets
train_data_dir = "asset/flower/train"
train_dir = "asset/working/train/"
test_dir = "asset/working/test/"


class ImageData(BaseModel):
    filename: str
    label: int | None
    descriptors: list
    histogram: list
    predict_label: int | None


def loadImages(root_folder: str) -> Tuple[List[ImageData], List[str]]:
    images = list()
    labels = set()

    for current_folder, subfolders, files in os.walk(root_folder):
        folder_label = os.path.basename(current_folder)
        if files:  # Only consider folders that contain files
            labels.add(folder_label)
    sorted_labels = sorted(labels)

    for current_folder, subfolders, files in os.walk(root_folder):
        folder_label = os.path.basename(current_folder)
        if files:
            label_index = sorted_labels.index(folder_label)
            for filename in files:
                file_path = os.path.join(current_folder, filename)
                image_matrix = readImage(file_path, 80, 80)
                if image_matrix is not None:
                    _, des = execSift(image_matrix)
                    image = ImageData(
                        filename=filename,
                        label=label_index,
                        descriptors=des,
                        histogram=None,
                        predict_label=None,
                    )
                    images.append(image)

    return images, sorted_labels


def kmeans(k, descriptors: list):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(descriptors)
    visual_words = kmeans.cluster_centers_
    return visual_words


def findIndex(feature, centers):
    distances = np.linalg.norm(centers - feature, axis=1)
    return np.argmin(distances)


def setHistograms(images: List[ImageData], centers: list) -> None:
    for image in images:
        histogram = np.zeros(len(centers))
        for each_feature in image.descriptors:
            ind = findIndex(each_feature, centers)
            histogram[ind] += 1
        image.histogram = histogram


def applyPcaToHistograms(histograms: list, n_components: int) -> list:
    pca = PCA(n_components=n_components)
    reduced_histograms = pca.fit_transform(histograms)
    return reduced_histograms


def knnWithPca(base_images: List[ImageData], target_images: List[ImageData]) -> None:
    sorted_labels = sorted({image.label for image in base_images})
    pvt_histograms_per_label = [list() for item in sorted_labels]
    for base in base_images:
        pvt_histograms_per_label[base.label].append(base.histogram)

    n_samples, n_features = np.array(pvt_histograms_per_label[0]).shape
    number_pca_components = min(n_samples, n_features)

    base_histograms_per_label = [list() for item in sorted_labels]
    for idx, histograms_per_label in enumerate(pvt_histograms_per_label):
        base_histograms_per_label[idx].append(
            applyPcaToHistograms(histograms_per_label, number_pca_components)
        )

    for target in target_images:
        target_histogram = applyPcaToHistograms(
            np.array(target.histogram).reshape(1, -1), number_pca_components
        ).flatten()
        minimum = distance.euclidean(target_histogram, base_histograms_per_label[0][0])
        key = target.label
        for idx, base_histogram in enumerate(base_histograms_per_label):
            dist = distance.euclidean(target_histogram, base_histogram)
            if dist < minimum:
                minimum = dist
                key = idx
        target.predict_label = key


def knn(base_images: List[ImageData], target_images: List[ImageData]) -> None:
    for target in target_images:
        minimum = distance.euclidean(target.histogram, base_images[0].histogram)
        key = target.label
        for base in base_images:
            dist = distance.euclidean(target.histogram, base.histogram)
            if dist < minimum:
                minimum = dist
                key = base.label
        target.predict_label = key


def main() -> None:
    # Data path
    data_dir = Path(train_data_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # Call the prepare_dataset function
    classes, train_df, test_df = prepare_dataset(data_dir, train_dir, test_dir)

    # Initialize new columns for descriptors and predict_label in train_df
    train_df["descriptors"] = None
    train_df["histogram"] = None
    train_df["predict_label"] = None

    # Iterate through each row to update the descriptors
    for idx, row in train_df.iterrows():
        image_matrix = readImage(row["image_path"], 80, 80)
        if image_matrix is not None:
            _, des = execSift(image_matrix)
            train_df.at[idx, "descriptors"] = des  # Update descriptors column

    # The `predict_label` column is already initialized with `None`, no further updates needed.

    print("Updated training DataFrame:")
    print(train_df.head())

    train_images, train_labels = loadImages(train_data_path)
    # print(f"ENUMS:\n{train_labels}")
    # print([(image.filename, image.label) for image in train_images])
    test_images, _ = loadImages(test_data_path)
    # print([(image.filename, image.label) for image in test_images])

    # Takes the central points which is visual words
    NUMBER_CENTROIDS = 150
    visual_words = kmeans(
        NUMBER_CENTROIDS, [image.descriptors for image in train_images]
    )
    # print(f"NUMBER OF CLUSTERS: {NUMBER_CENTROIDS}\nDATA:\n{visual_words}")

    # Creates histograms for train data
    setHistograms(train_images, visual_words)
    # Creates histograms for test data
    setHistograms(test_images, visual_words)

    # Call the knn function
    # knn(train_images, test_images)
    # knnWithPca(train_images, test_images)
    # print([(image.filename, image.predict_label) for image in test_images])

    # Calculates the accuracies and write the results to the console.
    knnWithPca(train_images, train_images)
    # knn(train_images, train_images)
    # print([(image.filename, image.label, image.predict_label) for image in train_images])
    good_predict = 0
    for image in train_images:
        if image.label == image.predict_label:
            good_predict += 1
    print(
        f"Acurracy: {(good_predict/len(test_images))*100}\nGood:{good_predict}\nBad:{len(test_images)-good_predict}"
    )


if __name__ == "__main__":
    main()
