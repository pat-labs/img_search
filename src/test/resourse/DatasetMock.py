import os
import random
import tempfile

import cv2 as cv
import numpy as np


class DatasetMock:
    CLASSES = ["cats", "dogs", "cars"]

    @staticmethod
    def _draw_shape_for_class(image, class_name):
        h, w, _ = image.shape
        center = (w // 2, h // 2)
        color_map = {
            "cats": (255, 0, 0),  # Blue
            "dogs": (0, 255, 0),  # Green
            "cars": (0, 0, 255)  # Red
        }
        color = color_map.get(class_name, (255, 255, 255))
        thickness = 4

        if class_name == "cats":
            cv.circle(image, center, 60, color, thickness)
            cv.line(image, (center[0] - 60, center[1]), (center[0] + 60, center[1]), color, 2)
        elif class_name == "dogs":
            cv.rectangle(image, (center[0] - 60, center[1] - 40), (center[0] + 60, center[1] + 40), color, thickness)
        elif class_name == "cars":
            pts = np.array([[center[0], center[1] - 60],
                            [center[0] - 60, center[1] + 60],
                            [center[0] + 60, center[1] + 60]], np.int32)
            cv.polylines(image, [pts], True, color, thickness)
        return image

    @staticmethod
    def _apply_variation(image):
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv.add(image, noise)
        h, w, _ = image.shape
        angle = random.uniform(-10, 10)
        M = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotated = cv.warpAffine(image, M, (w, h))

        if random.random() < 0.5:
            rotated = cv.GaussianBlur(rotated, (3, 3), 0)

        alpha = 1.0 + random.uniform(-0.2, 0.2)
        beta = random.uniform(-20, 20)
        adjusted = cv.convertScaleAbs(rotated, alpha=alpha, beta=beta)

        return adjusted

    @staticmethod
    def animals_mock(num_images_per_class: int = 3, img_size=(256, 256)):
        tmp_dir = tempfile.mkdtemp(prefix="img_search_")
        dataset_dir = os.path.join(tmp_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        random.seed(42)
        np.random.seed(42)

        for cls in DatasetMock.CLASSES:
            class_dir = os.path.join(dataset_dir, cls)
            os.makedirs(class_dir, exist_ok=True)

            for i in range(num_images_per_class):
                base = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * 255
                img = DatasetMock._draw_shape_for_class(base, cls)
                img = DatasetMock._apply_variation(img)

                filename = os.path.join(class_dir, f"{cls[:-1]}{i}.png")
                cv.imwrite(filename, img)

        print(f"Mock dataset created at: {tmp_dir}")
        return dataset_dir

    @staticmethod
    def get_mock_dir():
        return tempfile.mkdtemp(prefix="mock_")
