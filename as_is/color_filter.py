import os
from collections import Counter, defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from utils import save_plot_result


def extract_dominant_colors(image, k=5):
    """
    Extract dominant colors from an image using K-Means clustering.

    Parameters:
        image (numpy.ndarray): Input image (BGR format).
        k (int): Number of clusters (dominant colors to extract).

    Returns:
        dominant_colors (list): List of dominant colors in RGB format.
    """
    # Convert the image to Lab color space for better clustering
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = image_lab.reshape(-1, 3)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    dominant_colors_lab = kmeans.cluster_centers_

    # Convert dominant colors back to RGB for visualization
    dominant_colors_bgr = cv2.cvtColor(
        np.uint8([dominant_colors_lab]), cv2.COLOR_LAB2BGR
    )[0]
    dominant_colors_rgb = [
        tuple(map(int, color[::-1])) for color in dominant_colors_bgr
    ]  # BGR to RGB
    return dominant_colors_rgb


def analyze_dataset_by_color(dataset, k=5):
    """
    Analyze the dataset and find the most relevant colors for each class.

    Parameters:
        dataset (list): List of tuples (image_path, class_label).
        k (int): Number of clusters (dominant colors to extract per image).

    Returns:
        class_colors (dict): Dictionary with class labels as keys and most common colors as values.
    """
    class_color_map = defaultdict(list)

    for image_path, class_label in dataset:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            continue

        # Extract dominant colors
        dominant_colors = extract_dominant_colors(image, k=k)
        class_color_map[class_label].extend(dominant_colors)

    # Aggregate colors by class
    class_colors = {}
    for class_label, colors in class_color_map.items():
        # Count the most common colors for each class
        most_common_colors = Counter(colors).most_common(k)
        class_colors[class_label] = [color for color, _ in most_common_colors]

    return class_colors


def visualize_class_colors(class_colors):
    """
    Visualize the most relevant colors for each class.

    Parameters:
        class_colors (dict): Dictionary with class labels as keys and most common colors as values.
    """
    num_classes = len(class_colors)
    plt.figure(
        figsize=(8, 2 * num_classes)
    )  # Adjust height based on the number of classes

    for i, (class_label, colors) in enumerate(class_colors.items()):
        plt.subplot(num_classes, 1, i + 1)
        plt.title(f"Class: {class_label}")

        for j, color in enumerate(colors):
            # Normalize color values and convert to hexadecimal
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            plt.bar(j, 1, color=np.array(color) / 255, edgecolor="black", width=1)
            plt.text(
                j, 0.5, hex_color, ha="center", va="center", fontsize=10, color="black"
            )

        plt.xticks(range(len(colors)), [f"Color {j+1}" for j in range(len(colors))])
        plt.yticks([])

    plt.tight_layout()
    save_plot_result(plt, "colors_distribution.jpg")


def main():
    # Example dataset: List of (image_path, class_label)
    dataset = [
        ("asset/flower/train/daisy/5547758_eea9edfd54_n.jpg", "daisy"),
        ("asset/flower/train/daisy/99306615_739eb94b9e_m.jpg", "daisy"),
        ("asset/flower/train/rose/12240303_80d87f77a3_n.jpg", "rose"),
        ("asset/flower/train/rose/394990940_7af082cf8d_n.jpg", "rose"),
    ]

    # Analyze dataset and extract colors
    class_colors = analyze_dataset_by_color(dataset, k=3)

    # Visualize the colors for each class
    visualize_class_colors(class_colors)


if __name__ == "__main__":
    main()
