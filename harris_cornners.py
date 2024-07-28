import cv2
import matplotlib.pyplot as plt
import numpy as np


def harris_cornners(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # modify the data type
    # setting to 32-bit floating point
    operatedImage = np.float32(gray_img)

    # apply the cv2.cornerHarris method
    # to detect the corners with appropriate
    # values as input parameters
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

    # Results are marked through the dilated corners
    dest = cv2.dilate(dest, None)

    # Reverting back to the original image,
    # with optimal threshold value
    img[dest > 0.01 * dest.max()] = [0, 0, 255]

    # the window showing output image with corners
    cv2.imshow("Image with Borders", img)

    # De-allocate any associated memory usage
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()
