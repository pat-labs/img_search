import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, exposure
from skimage.feature import hog
from skimage.io import imread, imshow
from skimage.transform import resize


def hog_algorithm(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fd, hog_img = hog(
        gray_img,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
    )

    fd.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title("Input image")

    # Rescale histogram for better display
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))

    ax2.imshow(hog_img_rescaled, cmap=plt.cm.gray)
    ax2.set_title("Histogram of Oriented Gradients (HOG)")

    # store to file
    plt.savefig("result/img_hog.png", dpi=125)

    plt.show()


def calculate_gradient(img, template):
    ts = template.size  # Number of elements in the template (3).
    # New padded array to hold the resultant gradient image.
    new_img = np.zeros((img.shape[0] + ts - 1, img.shape[1] + ts - 1))
    new_img[
        np.uint16((ts - 1) / 2.0) : img.shape[0] + np.uint16((ts - 1) / 2.0),
        np.uint16((ts - 1) / 2.0) : img.shape[1] + np.uint16((ts - 1) / 2.0),
    ] = img
    result = np.zeros((new_img.shape))

    for r in np.uint16(np.arange((ts - 1) / 2.0, img.shape[0] + (ts - 1) / 2.0)):
        for c in np.uint16(np.arange((ts - 1) / 2.0, img.shape[1] + (ts - 1) / 2.0)):
            curr_region = new_img[
                r - np.uint16((ts - 1) / 2.0) : r + np.uint16((ts - 1) / 2.0) + 1,
                c - np.uint16((ts - 1) / 2.0) : c + np.uint16((ts - 1) / 2.0) + 1,
            ]
            curr_result = curr_region * template
            score = np.sum(curr_result)
            result[r, c] = score
    # Result of the same size as the original image after removing the padding.
    result_img = result[
        np.uint16((ts - 1) / 2.0) : result.shape[0] - np.uint16((ts - 1) / 2.0),
        np.uint16((ts - 1) / 2.0) : result.shape[1] - np.uint16((ts - 1) / 2.0),
    ]
    return result_img


def gradient_magnitude(horizontal_gradient, vertical_gradient):
    horizontal_gradient_square = np.power(horizontal_gradient, 2)
    vertical_gradient_square = np.power(vertical_gradient, 2)
    sum_squares = horizontal_gradient_square + vertical_gradient_square
    grad_magnitude = np.sqrt(sum_squares)
    return grad_magnitude


def gradient_direction(horizontal_gradient, vertical_gradient):
    grad_direction = np.arctan(vertical_gradient / (horizontal_gradient + 0.00000001))
    grad_direction = np.rad2deg(grad_direction)
    grad_direction = grad_direction % 180
    return grad_direction


def HOG_cell_histogram(cell_direction, cell_magnitude, hist_bins):
    HOG_cell_hist = np.zeros(shape=(hist_bins.size))
    cell_size = cell_direction.shape[0]

    for row_idx in range(cell_size):
        for col_idx in range(cell_size):
            curr_direction = cell_direction[row_idx, col_idx]
            curr_magnitude = cell_magnitude[row_idx, col_idx]

            diff = np.abs(curr_direction - hist_bins)

            if curr_direction < hist_bins[0]:
                first_bin_idx = 0
                second_bin_idx = hist_bins.size - 1
            elif curr_direction > hist_bins[-1]:
                first_bin_idx = hist_bins.size - 1
                second_bin_idx = 0
            else:
                first_bin_idx = np.where(diff == np.min(diff))[0][0]
                temp = hist_bins[
                    [
                        (first_bin_idx - 1) % hist_bins.size,
                        (first_bin_idx + 1) % hist_bins.size,
                    ]
                ]
                temp2 = np.abs(curr_direction - temp)
                res = np.where(temp2 == np.min(temp2))[0][0]
                if res == 0 and first_bin_idx != 0:
                    second_bin_idx = first_bin_idx - 1
                else:
                    second_bin_idx = first_bin_idx + 1

            first_bin_value = hist_bins[first_bin_idx]
            second_bin_value = hist_bins[second_bin_idx]
            HOG_cell_hist[first_bin_idx] = (
                HOG_cell_hist[first_bin_idx]
                + (np.abs(curr_direction - first_bin_value) / (180.0 / hist_bins.size))
                * curr_magnitude
            )
            HOG_cell_hist[second_bin_idx] = (
                HOG_cell_hist[second_bin_idx]
                + (np.abs(curr_direction - second_bin_value) / (180.0 / hist_bins.size))
                * curr_magnitude
            )
    return HOG_cell_hist


def hog_exec(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    horizontal_mask = np.array([-1, 0, 1])
    vertical_mask = np.array([[-1], [0], [1]])

    horizontal_gradient = calculate_gradient(gray_img, horizontal_mask)
    vertical_gradient = calculate_gradient(gray_img, vertical_mask)

    grad_magnitude = gradient_magnitude(horizontal_gradient, vertical_gradient)
    grad_direction = gradient_direction(horizontal_gradient, vertical_gradient)

    grad_direction = grad_direction % 180
    hist_bins = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170])

    # Histogram of the first cell in the first block.
    cell_direction = grad_direction[:8, :8]
    cell_magnitude = grad_magnitude[:8, :8]
    HOG_cell_hist = HOG_cell_histogram(cell_direction, cell_magnitude, hist_bins)

    plt.bar(x=np.arange(9), height=HOG_cell_hist, align="center", width=0.8)
    plt.show()
