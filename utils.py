import csv
import os
import pickle
import random
import time

import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def read_image(image_path):
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open {image_path}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray


def create_empty_file(file_path):
    with open(file_path, "w") as file:
        pass


def create_directory(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def save_to_csv(header, data, path):
    with open(path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(data)


def get_data_from_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Get the header (column names)
    header = df.columns.tolist()

    # Get the data
    data = df.values.tolist()

    return header, data


def load_gmm_model(model_path):
    with open(model_path, "rb") as f:
        gmm = pickle.load(f)
    return gmm


def save_keypoints(file_path, keypoints):
    if keypoints is None:
        print(f"TYPE: {type(keypoints)}\nKEYPOINTS:\n\t{keypoints}")
        raise ValueError("No keypoints")
    keypoints_list = [
        (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in keypoints
    ]
    save_to_pickle(file_path, keypoints_list)


def save_to_pickle(file_path, data):
    try:
        with open(file_path, "rb") as f:
            existing_data = pickle.load(f)
    except Exception as e:
        create_empty_file(file_path)
        existing_data = []

    existing_data.append(data)

    try:
        with open(file_path, "wb") as f:
            pickle.dump(existing_data, f)
        print(f"Saved to {file_path}.")
    except Exception as e:
        print(f"Error saving data: {e}")


def save_descriptors(file_path, descriptors):
    if descriptors is None:
        print(f"TYPE: {type(descriptors)}\nDESCRIPTORS:\n\t{descriptors}")
        raise ValueError("No descriptors")
    descriptors_list = descriptors.tolist()
    save_to_pickle(file_path, descriptors_list)


def load_descriptors(file_path):
    try:
        with open(file_path, "rb") as f:
            existing_data = pickle.load(f)
    except Exception as e:
        existing_data = []
    return np.array(existing_data)


def load_keypoints(file_path):
    data = np.load(file_path)

    keypoints_descriptors = []

    for keypoints_list in data:
        keypoints = [
            cv.KeyPoint(
                x=kp[0][0],
                y=kp[0][1],
                size=kp[1],
                angle=kp[2],
                response=kp[3],
                octave=kp[4],
                class_id=kp[5],
            )
            for kp in keypoints_list
        ]
        keypoints_descriptors.append(keypoints)

    return keypoints_descriptors


def time_it(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return result, elapsed_time


def set_plot(x_data, y_data, title, x_label, y_label, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort y_data based on x_data (if needed)
    sorted_indices = np.argsort(x_data)
    y_data_sorted = [y_data[i] for i in sorted_indices]
    x_data_sorted = [x_data[i] for i in sorted_indices]
    # Create the bar plot
    hbars = ax.barh(y_data_sorted, x_data_sorted, align="center")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Customize plot as needed (e.g., color, bar width, etc.)
    ax.grid(True)

    ax.bar_label(hbars, fmt="%.4f")
    # set right and top to slim
    max_value = max(x_data)
    space_factor = 0.09  # Adjust this factor as needed
    x_limit = max_value * (1 + space_factor)
    ax.set_xlim(right=x_limit)
    ax.spines["right"].set_color((0.8, 0.8, 0.8))
    ax.spines["top"].set_color((0.8, 0.8, 0.8))

    for bar in hbars:
        bar.set_facecolor([0.5, 0.5, 0.8, 0.3])
        bar.set_edgecolor([0, 0, 0.5, 0.3])
        bar.set_linewidth(2)

    x_lab = ax.xaxis.get_label()
    y_lab = ax.yaxis.get_label()
    x_lab.set_fontstyle("italic")
    x_lab.set_fontsize(10)
    y_lab.set_fontstyle("italic")
    y_lab.set_fontsize(10)
    ttl = ax.title
    ttl.set_fontweight("bold")

    plt.savefig(save_path)
    plt.close()


def set_plot_multiple_values(
    data, header, group_names, title, x_label, y_label, save_path
):
    fig, ax = plt.subplots(figsize=(12, 8))

    data = np.array(data)
    num_groups, num_bars = data.shape

    # Create an array for the x locations of the groups
    indices = np.arange(num_groups)

    # Set the width of the bars
    bar_width = 0.25

    face_colors = [[0.5, 0.5, 0.8, 0.3], [0.8, 0.5, 0.5, 0.3], [0.5, 0.8, 0.5, 0.3]]
    edge_color = [0, 0, 0.5, 0.3]
    line_width = 2

    # Plot bars in groups
    for i in range(num_bars):
        bars = ax.bar(
            indices + i * bar_width,
            data[:, i],
            bar_width,
            label=f"{header[i]}",
            color=[0.5, 0.5, 0.8, 0.3],
        )

        # Set custom properties for each bar
        for bar in bars:
            bar.set_facecolor(face_colors[i % len(face_colors)])
            bar.set_edgecolor(edge_color)
            bar.set_linewidth(line_width)

    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.spines["right"].set_color((0.8, 0.8, 0.8))
    ax.spines["top"].set_color((0.8, 0.8, 0.8))

    # Customize x-ticks with group names
    ax.set_xticks(indices + bar_width * (num_bars / 2 - 0.5))
    ax.set_xticklabels(group_names)

    # Add legend
    ax.legend()

    # Customize plot as needed
    ax.grid(True)

    # Adjust x-axis limit to accommodate all bars
    # max_value = data.max()
    # space_factor = 0.09  # Adjust this factor as needed
    # x_limit = max_value * (1 + space_factor)
    # ax.set_xlim(right=x_limit)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def increase_brightness(img_name, folder_path, destiny_path, brightness):
    img_path = os.path.join(folder_path, img_name)
    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"Could not open {img_path}")
    brighter = np.where(img + brightness > 255, 255, img)
    brighter_img_path = os.path.join(destiny_path, "brightness_" + img_name)
    cv.imwrite(brighter_img_path, brighter)


def rotate_image(img_name, folder_path, destiny_path, angle):
    img_path = os.path.join(folder_path, img_name)
    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"Could not open {img_path}")
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (w, h))
    rotated_img_path = os.path.join(destiny_path, "rotated_" + img_name)
    cv.imwrite(rotated_img_path, rotated)


def test_function():
    img_folder = "assets"
    img_name = "annapurna_1.jpg"
    increase_brightness(img_name, img_folder, img_folder, random.randint(1, 254))
    rotate_image(img_name, img_folder, img_folder, random.randint(1, 359))


def apply_transform():
    img_folder = "dataset//flowers//train//sunflower"
    brightness_folder = "dataset//flowers//brightness"
    rotation_folder = "dataset//flowers//rotated"

    if not os.path.exists(brightness_folder):
        os.makedirs(brightness_folder)
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        bright_img_path = os.path.join(brightness_folder, img_name)
        increase_brightness(img_path, bright_img_path)

    if not os.path.exists(rotation_folder):
        os.makedirs(rotation_folder)
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        rotated_img_path = os.path.join(rotation_folder, img_name)
        rotate_image(img_path, rotated_img_path)


if __name__ == "__main__":
    test_function()
