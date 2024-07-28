import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def generate_sunflower_matrix(img):
    if img is None:
        raise ValueError("Image not loaded properly. Please check the path and file.")

    img //= 255  # Now the pixels have range [0, 1]
    img_list = img.tolist()  # We have a list of lists of pixels

    result = ""
    for row in img_list:
        row_str = [str(p) for p in row]
        result += "[" + ", ".join(row_str) + "],\n"

    return result


def detect_circles(
    gray, dp=1.2, min_dist=50, param1=50, param2=30, min_radius=0, max_radius=0
):
    # Apply a Gaussian blur to reduce noise and improve detection
    blurred = cv.GaussianBlur(gray, (9, 9), 2)

    # Use HoughCircles to detect circles
    circles = cv.HoughCircles(
        blurred,
        cv.HOUGH_GRADIENT,
        dp,
        min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    # If some circles are detected, convert the (x, y) coordinates and radius to integers
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Draw the circles on the original image
        for x, y, r in circles:
            cv.circle(gray, (x, y), r, (0, 255, 0), 4)
            cv.rectangle(gray, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    return gray, circles


if __name__ == "__main__":
    img_path = "C://Users//pa-tr//Documents//projects//img_search//assets//sunflower_shape_draw.png"
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    stretch_near = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)

    # matrix_string = generate_sunflower_matrix(stretch_near)
    # print(matrix_string)

    detected_image, circles = detect_circles(stretch_near)
    # Display the output image with detected circles
    cv.imshow("Detected Circles", detected_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
