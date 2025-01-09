import os
import pickle
from typing import List, Tuple, Union

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pydantic import BaseModel
from skimage.feature import fisher_vector, learn_gmm
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC

from algorithms_keypoints_descriptors import execSift, readImage


class ImageData(BaseModel):
    filename: str
    label: int | None
    descriptors: list
    predict_label: int | None


result_path = "result/"


def loadImages(root_folder: str) -> Tuple[List[ImageData], List[str]]:
    images = list()
    labels = set()

    for current_folder, subfolders, files in os.walk(root_folder):
        folder_label = os.path.basename(current_folder)
        if files:
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
                        predict_label=None,
                    )
                    images.append(image)

    return images, sorted_labels


def makeReportModel(
    filename, best_params, best_estimator, cross_val_accuracy, report_dict
):
    # Generate Markdown for the configuration table
    markdown_config = f"""
    #### **Model Configuration**

    | Metric              | Value                                  |
    |---------------------|----------------------------------------|
    | Best Parameters      | `{{k: v for k, v in best_params.items()}}` |
    | Best Estimator       | `{best_estimator}`                    |
    | Cross-Val Accuracy   | `{cross_val_accuracy}`                |
    """

    # Generate Markdown for the classification report table
    markdown_report = """
    #### **Classification Report**

    | Class | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|---------|
    """

    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            markdown_report += f"| {label} | {metrics['precision']} | {metrics['recall']} | {metrics['f1-score']} | {metrics['support']} |\n"
        else:
            markdown_report += f"| **Accuracy** |        |        | **{metrics}**  | {report_dict['macro avg']['support']} |\n"

    markdown_report += f"| **Macro Avg** | {report_dict['macro avg']['precision']} | {report_dict['macro avg']['recall']} | {report_dict['macro avg']['f1-score']} | {report_dict['macro avg']['support']} |\n"
    markdown_report += f"| **Weighted Avg** | {report_dict['weighted avg']['precision']} | {report_dict['weighted avg']['recall']} | {report_dict['weighted avg']['f1-score']} | {report_dict['weighted avg']['support']} |\n"

    # Combine both markdown tables
    final_markdown = markdown_config + markdown_report

    with open(result_path + filename + ".md", "w") as file:
        file.write(final_markdown)
    # print(final_markdown)


def trainModel(
    train_descriptors, test_descriptors, train_targets, test_targets, filename
):
    # k_nodes = 16
    # k_nodes = 32
    k_nodes = 64

    gmm = learn_gmm(train_descriptors, n_modes=k_nodes)

    # Compute the Fisher vectors
    training_fvs = np.array(
        [fisher_vector(descriptor_mat, gmm) for descriptor_mat in train_descriptors]
    )
    testing_fvs = np.array(
        [fisher_vector(descriptor_mat, gmm) for descriptor_mat in test_descriptors]
    )

    # 1. Advanced Fisher Vector Encoding Techniques
    # training_fvs = np.sign(training_fvs) * np.sqrt(np.abs(training_fvs))
    # testing_fvs = np.sign(testing_fvs) * np.sqrt(np.abs(testing_fvs))
    # 2.Normalize Fisher Vectors
    # training_fvs = normalize(training_fvs, norm='l2')
    # testing_fvs = normalize(testing_fvs, norm='l2')
    # 3.Apply PCA
    # n_samples, n_features = np.array(training_fvs).shape
    # n_components = min(n_samples, n_features)
    # pca = PCA(n_components=n_components)
    # training_fvs = pca.fit_transform(training_fvs)
    # testing_fvs = pca.transform(testing_fvs)

    # Hyperparameter Tuning
    # 1.
    # param_grid = {
    #    "C": [0.1, 1, 10, 100],
    #    "gamma": [1e-3, 1e-4, "scale"],
    #    "kernel": ["rbf"],
    # }
    # grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    # 2.
    # param_grid = {'degree': [2, 3, 4], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'coef0': [0.0, 0.5, 1.0]}
    # grid = GridSearchCV(SVC(kernel='poly'), param_grid, refit=True, verbose=3)
    # 3.
    param_grid = {"C": [0.01, 0.1, 1, 10, 100], "max_iter": [1000, 5000, 10000]}
    grid = GridSearchCV(LinearSVC(), param_grid, cv=5, verbose=3, n_jobs=-1)

    grid.fit(training_fvs, train_targets)
    best_svm = grid.best_estimator_
    best_model_scores = cross_val_score(
        best_svm, training_fvs, train_targets, cv=5, n_jobs=-1
    )
    predictions = best_svm.predict(testing_fvs)

    # df = pd.DataFrame({'Prediction': predictions})
    # df.to_csv('result/data/predictions.csv', index=False)

    report_dict = classification_report(test_targets, predictions, output_dict=True)
    makeReportModel(
        filename,
        grid.best_params_,
        grid.best_estimator_,
        np.mean(best_model_scores),
        report_dict,
    )

    ConfusionMatrixDisplay.from_estimator(
        best_svm,
        testing_fvs,
        test_targets,
        cmap=plt.cm.Blues,
    )
    plt.savefig(result_path + filename + ".jpg")

    return gmm


def getFisherVector(descriptor, gmm):
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    N = descriptor.shape[0]
    K = means.shape[0]

    # Compute posterior probabilities
    q = gmm.predict_proba(descriptor)

    # Compute Fisher vectors
    fisher_vector = np.zeros(2 * K * descriptor.shape[1])

    for i in range(K):
        # Calculate gradients with respect to mean
        diff = descriptor - means[i]
        fisher_vector[i * descriptor.shape[1] : (i + 1) * descriptor.shape[1]] = np.sum(
            q[:, i][:, np.newaxis] * diff / np.sqrt(covariances[i]), axis=0
        )

        # Calculate gradients with respect to covariance
        fisher_vector[
            (K + i) * descriptor.shape[1] : (K + i + 1) * descriptor.shape[1]
        ] = np.sum(
            q[:, i][:, np.newaxis] * (diff**2 - covariances[i]) / (2 * covariances[i]),
            axis=0,
        )

    # Normalize Fisher vector
    fisher_vector = normalize(fisher_vector.reshape(1, -1))[0]

    return fisher_vector


if __name__ == "__main__":
    train_images, train_labels = loadImages("asset/flower/train")
    # print(f"ENUMS:\n{train_labels}")
    # print([(image.filename, image.label) for image in train_images])
    # Split the data into training and testing subsets
    descriptors = list()
    targets = list()
    for image in train_images:
        descriptors.append(image.descriptors)
        targets.append(image.label)
    train_descriptors, test_descriptors, train_targets, test_targets = train_test_split(
        descriptors, targets
    )

    # filename = "k_nodes_16_hyperparameter_3"
    filename = "the_choosen_one_k_nodes_64_hyperparameter_3"
    gmm = trainModel(
        train_descriptors,
        test_descriptors,
        train_targets,
        test_targets,
        filename,
    )
    with open(result_path + filename + ".pkl", "wb") as f:
        pickle.dump(gmm, f)
