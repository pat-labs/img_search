import os
from collections import Counter

import numpy as np
from PIL import Image


def extract_palette_rgb(image_path, num_colors=5):
    """Extracts a color palette from an image.

    Args:
      image_path: Path to the image file.
      num_colors: Number of colors in the palette.

    Returns:
      A list of (R, G, B) color tuples.
    """

    # Load the image
    img = Image.open(image_path)

    # Resize the image for faster processing (optional)
    img = img.resize((200, 200))

    # Convert to RGB mode
    img = img.convert("RGB")

    # Convert image to NumPy array
    img_array = np.array(img)

    # Reshape the array to a flat list of pixels
    pixels = img_array.reshape(-1, 3)

    # Quantize the colors (optional)
    # You can use libraries like k-means clustering or other quantization techniques

    # Count the occurrences of each color
    color_counts = Counter(map(tuple, pixels))

    # Get the most common colors
    palette = [color for color, count in color_counts.most_common(num_colors)]

    return palette


def rgb_to_hex(rgb):
    hex_color = "#" + "".join(f"{component:02x}" for component in rgb)
    return hex_color


if __name__ == "__main__":
    img_folder = "C://Users//pa-tr//Documents//projects//img_search//dataset//flowers//train//sunflower"

    test_image = os.path.join(img_folder, "5955500463_6c08cb199e.jpg")
    palette = extract_palette_rgb(test_image, num_colors=10)
    print(palette)

    hex_color = rgb_to_hex(palette[0])
    print(hex_color)
    # for img_name in os.listdir(img_folder):
    #    img_path = os.path.join(img_folder, img_name)
