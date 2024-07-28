import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance
from skimage.feature import hog
from skimage.io import imread


def load_images(path):
    # read the images and store in a list
    images = [imread(file) for file in glob.glob(path + "/*.jpg")]

    # number of images
    n = len(images)

    # show images

    fig = plt.figure(figsize=(16, 8))

    for i in range(n):
        fig.add_subplot(2, 4, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.title(i)

    plt.show()
    return images


def get_hog_features(images):
    # creating a list to store HOG feature vectors
    fd_list = []

    fig = plt.figure(figsize=(12, 12))
    k = 0

    for i in range(n):

        # execute hog function for each image that is imported from skimage.feature module
        fd, hog_image = hog(
            images[i],
            orientations=9,
            pixels_per_cell=(64, 64),
            cells_per_block=(2, 2),
            visualize=True,
            multichannel=True,
        )

        # add the feature vector to the list
        fd_list.append(fd)

        # display hog image
        fig.add_subplot(4, 4, k + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.title(i)

        # display original image
        fig.add_subplot(4, 4, k + 2)
        plt.imshow(hog_image)
        plt.axis("off")

        k += 2

    plt.show()


def get_distances(fd_list):
    # create an empty nxn distance matrix
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        fd_i = fd_list[i]
        for k in range(i):
            fd_k = fd_list[k]
            # measure Jensen–Shannon distance between each feature vector
            # and add to the distance matrix
            distance_matrix[i, k] = distance.jensenshannon(fd_i, fd_k)

    # symmetrize the matrix as distance matrix is symmetric
    distance_matrix = np.maximum(distance_matrix, distance_matrix.transpose())

    print(distance_matrix)

    # convert square-form distance matrix to vector-form distance vector (condensed distance matrix)
    cond_distance_matrix = distance.squareform(distance_matrix)

    print(cond_distance_matrix)

    Z = linkage(cond_distance_matrix, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(Z, color_threshold=0.2, show_leaf_counts=True)
    plt.show()
