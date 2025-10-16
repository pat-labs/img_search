import cv2
import numpy as np


class ImageSanitizer:
    @staticmethod
    def noise_reduction(image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    @staticmethod
    def normalize(image):
        norm_img = np.zeros_like(image, dtype=np.uint8)
        return cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

    @staticmethod
    def contrast_enhancement(image):
        if len(image.shape) == 3 and image.shape[2] == 3:  # HSV image
            h, s, v = cv2.split(image)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_v = clahe.apply(v)
            enhanced_hsv = cv2.merge([h, s, enhanced_v])
            return enhanced_hsv
        else:  # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    @staticmethod
    def binarize(image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        return binary_image

    def sanitize(img: np.array, image_size=(224, 224)):
        try:
            resized_img = cv2.resize(img, image_size)
            denoised_img = self.noise_reduction(resized_img)
            hsv_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2HSV)
            enhanced_img = self.contrast_enhancement(hsv_img)
            normalized_img = self.normalize(enhanced_img)

            target_class_dir = os.path.join(sanitized_dataset_type_path, class_label)
            if not os.path.exists(target_class_dir):
                os.makedirs(target_class_dir)

            image_name = os.path.basename(image_path)
            sanitized_image_path = os.path.join(target_class_dir, image_name)
            cv2.imwrite(sanitized_image_path, normalized_img)
        except Exception as e:
            print(f"Error processing image {img}: {e}")