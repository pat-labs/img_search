import cv2 as cv
import numpy as np
from skimage.transform import resize


def readImage(img_path: str, width: int, height: int):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not open {img_path}")

    image_resized = resize(img, (width, height))
    # Convertir la imagen a uint8 (rango [0, 255])
    image_resized = (image_resized * 255).astype(np.uint8)

    return image_resized


def drawKeypoints(img, destiny_path: str, kp):
    img = cv.drawKeypoints(
        img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv.imwrite(destiny_path, img)


def increaseBrightness(img_path, destiny_path, brightness):
    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"Could not open {img_path}")
    brighter = np.where(img + brightness > 255, 255, img)
    cv.imwrite(destiny_path, brighter)


def rotateImage(img_path, destiny_path, angle):
    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"Could not open {img_path}")
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (w, h))
    cv.imwrite(destiny_path, rotated)


def rotatedMatch(kp1, des1, kp2, des2):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    p_matched = []
    dist_list = []
    distance = []

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    p_matched.append(len(matches) / len(kp1))

    for match in matches:
        c1 = kp1[match.queryIdx].pt
        c2 = kp2[match.trainIdx].pt
        dist = (
            (c1[0] - (c2[0] * (-1) + 720)) ** 2 + (c1[1] - (c2[1] * (-1) + 480)) ** 2
        ) ** 0.5
        distance.append(dist)
    dist_list.append(np.average(dist))
    per_r = np.average(p_matched)
    dist_r = np.average(dist_list)

    return per_r, dist_r


def brightnessMatch(kp1, des1, kp2, des2):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    p_matched = []
    dist_list = []
    distance = []

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    p_matched.append(len(matches) / len(kp1))

    for match in matches:
        c1 = kp1[match.queryIdx].pt
        c2 = kp2[match.trainIdx].pt
        dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
        distance.append(dist)
    dist_list.append(np.average(dist))
    per_r = np.average(p_matched)
    dist_r = np.average(dist_list)

    return per_r, dist_r


def increase_contrast(img):
    # converting to LAB color space
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    return np.hstack((img, enhanced_img))


def execSift(img, features: int = 5000):
    sift = cv.SIFT_create(features)
    kp, des = sift.detectAndCompute(img, None)

    return (kp, des)


def execSurf(img):
    surf = cv.xfeatures2d_SURF.create()
    kp, des = surf.detectAndCompute(img, None)

    return (kp, des)


def execOrb(img, features: int = 5000):
    orb = cv.ORB_create(features)
    kp, des = orb.detectAndCompute(img, None)

    return (kp, des)


def execKaze(img):
    kaze = cv.KAZE.create()
    kp, des = kaze.detectAndCompute(img, None)

    return (kp, des)


def execAkaze(img):
    akaze = cv.AKAZE_create()
    kp, des = akaze.detectAndCompute(img, None)

    return (kp, des)


def execBrisk(img):
    brisk = cv.BRISK_create()
    kp, des = brisk.detectAndCompute(img, None)

    return (kp, des)


def execSiftFreak(img):
    sift = cv.SIFT_create()
    kp = sift.detect(img, None)

    freak = cv.xfeatures2d.FREAK_create()
    kp, des = freak.compute(img, kp)

    return (kp, des)


def execStartBrief(img, desired_keypoints: int = 5000):
    max_size = 45
    response_threshold = 30
    line_threshold_projected = 10
    line_threshold_binarized = 8
    suppress_nonmax_size = 5

    while True:
        star = cv.xfeatures2d.StarDetector_create(
            maxSize=max_size,
            responseThreshold=response_threshold,
            lineThresholdProjected=line_threshold_projected,
            lineThresholdBinarized=line_threshold_binarized,
            suppressNonmaxSize=suppress_nonmax_size,
        )
        kp = star.detect(img, None)

        if len(kp) >= desired_keypoints:
            break

        # Adjust parameters based on your desired goal
        response_threshold -= 5  # Example: decrease responseThreshold

    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp, des = brief.compute(img, kp)

    return (kp, des)
