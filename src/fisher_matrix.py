import json
import os
import pickle
import time
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pydantic import BaseModel
from skimage.feature import fisher_vector, learn_gmm
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC

from algorithms_keypoints_descriptors import execSift, readImage
from utils import prepare_dataset, save_plot_result

# Create directories for training and testing datasets
train_data_dir = "asset/flower/train"
train_dir = "asset/working/train/"
test_dir = "asset/working/test/"


confusion_matrix_path = "result/plot"
result_model_path = "model"
model_report_path = "result/data"


class ImageData(BaseModel):
    filename: str
    label: int | None
    descriptors: list | None
    predict_label: int | None


class KNodes(Enum):
    # K16 = 16
    # K32 = 32
    # K64 = 64
    K128 = 128
    # K256 = 256


class EncodingTechnique(Enum):
    NO_ENCODING = 1
    # ADVANCED_FISHER = 2
    NORMALIZE_FISHER = 2
    APPLY_PCA = 3


class HyperparameterTechnique(Enum):
    #SVM_RBF = 1
    #SVM_POLY = 2
    LINEAR_SVC = 1


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


def makeReportJSON(
    filename, best_params, best_estimator, cross_val_accuracy, report_dict
):
    # Serialize only the best estimator's parameters, not the full object
    if hasattr(best_estimator, "get_params"):
        best_estimator_params = best_estimator.get_params()
    else:
        best_estimator_params = str(best_estimator)  # Fallback to string representation

    report_json = {
        "model_configuration": {
            "best_parameters": best_params,
            "best_estimator": best_estimator_params,  # Use parameters or a string
            "cross_val_accuracy": cross_val_accuracy,
        },
        "classification_report": {"metrics": report_dict},
    }

    # Save the JSON report to a file
    with open(f"{model_report_path}/{filename}.json", "w") as file:
        json.dump(report_json, file, indent=4)


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

    with open(f"{model_report_path}/{filename}.md", "w") as file:
        file.write(final_markdown)
    # print(final_markdown)


def trainModel(
    k_nodes,
    grid,
    train_descriptors,
    test_descriptors,
    train_labels,
    test_labels,
    filename,
):
    gmm = learn_gmm(train_descriptors, n_modes=k_nodes)

    # Compute the Fisher vectors
    training_fvs = np.array(
        [
            fisher_vector(train_descriptor_mat, gmm)
            for train_descriptor_mat in train_descriptors
        ]
    )
    testing_fvs = np.array(
        [
            fisher_vector(test_descriptor_mat, gmm)
            for test_descriptor_mat in test_descriptors
        ]
    )

    grid.fit(training_fvs, train_labels)
    best_svm = grid.best_estimator_
    best_model_scores = cross_val_score(
        best_svm, training_fvs, train_labels, cv=5, n_jobs=-1
    )
    predictions = best_svm.predict(testing_fvs)

    # df = pd.DataFrame({'Prediction': predictions})
    # df.to_csv('result/data/predictions.csv', index=False)

    report_dict = classification_report(test_labels, predictions, output_dict=True)
    makeReportJSON(
        filename,
        grid.best_params_,
        grid.best_estimator_,
        np.mean(best_model_scores),
        report_dict,
    )

    ConfusionMatrixDisplay.from_estimator(
        best_svm,
        testing_fvs,
        test_labels,
        cmap=plt.cm.Blues,
    )

    plt.savefig(f"{confusion_matrix_path}/{filename}.jpg")
    with open(f"{result_model_path}/{filename}.pkl", "wb") as f:
        pickle.dump(gmm, f)

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


def apply_encoding(encoding_type, training_fvs, testing_fvs):
    if encoding_type == EncodingTechnique.NO_ENCODING:
        pass  # No encoding applied.
    # elif encoding_type == EncodingTechnique.ADVANCED_FISHER:
    #     # 1. Advanced Fisher Vector Encoding Techniques
    #     training_fvs = np.sign(training_fvs) * np.sqrt(np.abs(training_fvs))
    #     testing_fvs = np.sign(testing_fvs) * np.sqrt(np.abs(testing_fvs))
    elif encoding_type == EncodingTechnique.NORMALIZE_FISHER:
        # 2. Normalize Fisher Vectors
        training_fvs = normalize(training_fvs, norm="l2")
        testing_fvs = normalize(testing_fvs, norm="l2")
    elif encoding_type == EncodingTechnique.APPLY_PCA:
        # 3. Apply PCA
        training_fvs = np.array(training_fvs)
        testing_fvs = np.array(testing_fvs)

        n_samples, n_features = training_fvs.shape
        n_components = min(n_samples, n_features)

        pca = PCA(n_components=n_components)
        training_fvs = pca.fit_transform(training_fvs)
        testing_fvs = pca.transform(testing_fvs)
    else:
        raise ValueError("Invalid encoding_type. Choose a valid EncodingTechnique.")

    return training_fvs, testing_fvs


def hyperparameter_technique(hyperparameter_type):
    # if hyperparameter_type == HyperparameterTechnique.SVM_RBF:
    #     # Hyperparameter tuning for SVM with RBF kernel
    #     param_grid = {
    #         "C": [0.1, 1, 10, 100],
    #         "gamma": [1e-3, 1e-4, "scale"],
    #         "kernel": ["rbf"],
    #     }
    #     grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    # elif hyperparameter_type == HyperparameterTechnique.SVM_POLY:
    #     # Hyperparameter tuning for SVM with polynomial kernel
    #     param_grid = {
    #         "degree": [2, 3, 4],
    #         "C": [0.1, 1, 10],
    #         "gamma": ["scale", "auto"],
    #         "coef0": [0.0, 0.5, 1.0],
    #     }
    #     grid = GridSearchCV(SVC(kernel="poly"), param_grid, refit=True, verbose=3)
    # elif 
    if hyperparameter_type == HyperparameterTechnique.LINEAR_SVC:
        # Hyperparameter tuning for Linear SVC
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "max_iter": [1000, 5000, 10000],
        }
        grid = GridSearchCV(LinearSVC(), param_grid, cv=5, verbose=3, n_jobs=-1)
    else:
        raise ValueError(
            "Invalid hyperparameter_type. Choose a valid HyperparameterTechnique."
        )

    return grid


def display_probabilities(gmm, data, labels, new_data_point, probabilities, filename):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter (gmm. means [:, 0], gmm.means [:, 1], c='red', marker='X')
    plt.scatter(new_data_point[:, 0], new_data_point[:, 1], c='black', marker='o', label='New Point')
    plt.title("Gaussian Mixture Model Clustering with New Data Point")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2") 
    plt.legend()
    save_plot_result(plt, filename)
    print("Probabilities of the new data point belonging to each cluster:", probabilities)

def main():
    # Data path
    data_dir = Path(train_data_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # Call the prepare_dataset function
    classes, train_df, test_df = prepare_dataset(data_dir, train_dir, test_dir)

    # Initialize new columns for descriptors and predict_label in train_df
    train_df["descriptors"] = None
    train_df["predict_label"] = None
    train_rows_to_drop = []

    # Iterate through each row to update the descriptors
    for idx, row in train_df.iterrows():
        image_matrix = readImage(row["image_path"], 80, 80)
        if image_matrix is not None:
            _, des = execSift(image_matrix)
            if des is None:
                train_rows_to_drop.append(idx)
                print(f"Descriptor is empty for the file: {row['image_path']}")
                continue
            train_df.at[idx, "descriptors"] = des
    # Drop rows with empty descriptors
    train_df.drop(index=train_rows_to_drop, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    test_df["descriptors"] = None
    test_df["predict_label"] = None
    test_rows_to_drop = []
    # Iterate through each row to update the descriptors
    for idx, row in test_df.iterrows():
        image_matrix = readImage(row["image_path"], 80, 80)
        if image_matrix is not None:
            _, des = execSift(image_matrix)
            if des is None:
                test_rows_to_drop.append(idx)
                print(f"Descriptor is empty for the file: {row['image_path']}")
                continue
            test_df.at[idx, "descriptors"] = des
    # Drop rows with empty descriptors
    test_df.drop(index=test_rows_to_drop, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_descriptors = list(train_df["descriptors"])
    train_label = list(train_df["label"])
    test_descriptors = list(test_df["descriptors"])
    test_label = list(test_df["label"])
    performance = {}

    for k_node in KNodes:
        #for encoding in EncodingTechnique:
        for hyperparameter in HyperparameterTechnique:
            # Start time measurement
            start_time = time.time()

            # Create a clean and readable filename
            filename = (
                f"fisher_matrix_k{str(k_node.value)}_"
                #f"encoding_{encoding.name.lower()}_"
                f"hyperparameter_{hyperparameter.name.lower()}.model"
            )
            print(f"Title: {filename}")

            # Perform encoding
            # train_fvs, test_fvs = apply_encoding(
            #     encoding, train_descriptors, test_descriptors
            # )

            # Perform hyperparameter tuning
            grid = hyperparameter_technique(hyperparameter)

            # Train the model
            gmm = trainModel(
                k_node.value,
                grid,
                train_descriptors,
                test_descriptors,
                train_label,
                test_label,
                filename,
            )

            # End time measurement
            end_time = time.time()
            time_lapse = end_time - start_time

            # Update performance dictionary
            performance.update({"model": filename, "time_lapse": time_lapse})
            print(f"Model: {filename}, Time Lapse: {time_lapse:.2f} seconds")


if __name__ == "__main__":
    main()
