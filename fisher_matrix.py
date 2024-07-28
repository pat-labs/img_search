import os
import pickle
import random

import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import ORB, fisher_vector, learn_gmm
from skimage.transform import resize
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from utils import create_empty_file, read_image


def get_dataset(dataset_train_path, dataset_test_path):
    train = list()
    for root, _, files in os.walk(dataset_train_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                image_path = os.path.join(root, file)
                _, gray = read_image(image_path)
                if gray is not None:
                    train.append(gray)

    target = list()
    for root, _, files in os.walk(dataset_test_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                image_path = os.path.join(root, file)
                _, gray = read_image(image_path)
                if gray is not None:
                    target.append(gray)

    # Equalize lengths by sampling from the larger list
    if len(train) > len(target):
        train = random.sample(train, len(target))
    elif len(target) > len(train):
        target = random.sample(target, len(train))

    return train, target


def extract_descriptors(images):
    # Resize images so that ORB detects interest points for all images
    images = np.array([resize(image, (80, 80)) for image in images])

    # Compute ORB descriptors for each image
    descriptors = []
    for image in images:
        detector_extractor = ORB(n_keypoints=5, harris_k=0.01)
        detector_extractor.detect_and_extract(image)
        descriptors.append(detector_extractor.descriptors.astype("float32"))

    return descriptors


def test(images, targets, plot_path):
    image_descriptors = extract_descriptors(images)
    target_descriptors = extract_descriptors(targets)

    # Flatten the list of descriptors
    image_descriptors = np.vstack(image_descriptors)
    target_descriptors = np.vstack(target_descriptors)

    # Create labels for each descriptor
    image_labels = np.zeros(len(image_descriptors))
    target_labels = np.ones(len(target_descriptors))

    # Combine the data
    descriptors = np.vstack([image_descriptors, target_descriptors])
    labels = np.hstack([image_labels, target_labels])

    # Split the data into training and testing subsets
    train_descriptors, test_descriptors, train_targets, test_targets = train_test_split(
        descriptors, labels, test_size=0.2, random_state=42
    )

    # Train a K-mode GMM
    k = 16
    gmm = learn_gmm(train_descriptors, n_modes=k)

    # Compute the Fisher vectors
    training_fvs = np.array(
        [
            fisher_vector(descriptor_mat.reshape(1, -1), gmm)
            for descriptor_mat in train_descriptors
        ]
    )

    testing_fvs = np.array(
        [
            fisher_vector(descriptor_mat.reshape(1, -1), gmm)
            for descriptor_mat in test_descriptors
        ]
    )

    svm = LinearSVC(max_iter=10000).fit(training_fvs, train_targets)

    predictions = svm.predict(testing_fvs)

    print(classification_report(test_targets, predictions))

    ConfusionMatrixDisplay.from_estimator(
        svm,
        testing_fvs,
        test_targets,
        cmap=plt.cm.Blues,
    )

    plt.savefig(plot_path)
    plt.show()

    return gmm


if __name__ == "__main__":
    dataset_train_path = "C://Users//pa-tr//Documents//projects//img_search//dataset//flowers//train//sunflower"
    dataset_test_path = (
        "C://Users//pa-tr//Documents//projects//img_search//dataset//flowers//test"
    )
    model_path = "C://Users//pa-tr//Documents//projects//img_search//result//model//gmm_model_fisher_vector.pkl"
    plot_path = "C://Users//pa-tr//Documents//projects//img_search//result//plot//confusion_matrix.png"
    train, targets = get_dataset(dataset_train_path, dataset_test_path)
    gmm = test(train, targets, plot_path)
    # Save the GMM model to a file
    with open(model_path, "wb") as f:
        pickle.dump(gmm, f)
