import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import requests


#
def hsl():
    pass


def splitting_merging():
    pass


# Keypoints detectors


def agast():
    pass


def start():
    pass


def read_image_from_url(url):
    """Reads an image from the specified URL and returns a NumPy array.

    Args:
        url: The URL of the image to download.

    Returns:
        A NumPy array representing the image, or None if an error occurs.
    """

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for non-200 status codes

        # Handle potential content-type issues (e.g., not an image)
        # if not response.headers['Content-Type'].startswith('image/'):
        #   print(f"Error: URL '{url}' does not appear to contain an image.")
        #   return None

        # Load the image data from the response
        img_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode image for OpenCV

        return img
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from URL '{url}': {e}")
        return None


if __name__ == "__main__":
    # Example usage
    img1_url = "https://res.cloudinary.com/hello-tickets/image/upload/c_limit,f_auto,q_auto,w_768/v1612435917/post_images/rome-124/michele-bitetto-fontana-di-trevi.jpg"
    img2_url = "https://www.turismoroma.it/sites/default/files/Fontane%20-%20Fontana%20di%20Trevi_1920x1080mba-07410189%20%C2%A9%20Clickalps%20_%20AGF%20foto.jpg"
    img3_url = "https://www.vintagevolksweddings.co.uk/images/vw-camper-wedding-car-sheffield-front.jpg"
    img1 = read_image_from_url(img1_url)
    img2 = read_image_from_url(img2_url)
    img3 = read_image_from_url(img3_url)
    # sift_algorithm(img1)
    # orb_algorithm(img1)
    # sift_matches(img1, img2)
    hog_algorithm(img3)
