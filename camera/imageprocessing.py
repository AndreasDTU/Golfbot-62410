from unittest import result

import cv2
import numpy as np


lower = np.array([0, 0, 0])        # lower bound (adjust H/S if needed)
upper = np.array([180, 205, 210])    # upper bound using your values as upper threshold

lower_white = np.array([0, 0, 240])
upper_white = np.array([180, 40, 255])

lower_orange = np.array([0, 180, 150])
upper_orange = np.array([60, 255, 240])

pallete = {
    "Floor" : [0, 0, 0],
    "Ball" : [180, 105, 255],
    "Orange Ball" : [255, 255, 0],
    "Robot" : [255, 0, 255],
    "Wall" : [255, 255, 255]
}

def distsq(pixel, color_value):
    return sum((a - b) ** 2 for a, b in zip(pixel, color_value))


def mapToNearestColor(pixel):
    min_dist = float('inf')
    nearest_color = None
    for color_name, color_value in pallete.items():
        dist = distsq(pixel, color_value)
        if dist < min_dist:
            min_dist = dist
            nearest_color = color_name
    return nearest_color

def imageprocessing(img, colorspace):
    hsv = cv2.cvtColor(img, colorspace)
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(img, img, mask=mask_inv)
    cv2.imshow('Original colors of detected objects', result)
    cv2.imwrite('colored_result.png', result)
    #Masks
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    copy = result.copy()

    copy[mask_white > 0] = [180, 105, 255]
    copy[mask_orange > 0] = [255, 255, 0]
    
    cv2.imshow('Result', copy)
    cv2.imwrite('colored_balls.png', copy)

    return result


def undistort_with_calibration(img, calibration_file, balance=0.0):
    data = np.load(calibration_file)
    K = data["K"]
    D = data["D"]
    calibration_image_size = tuple(int(v) for v in data["image_size"])

    height, width = img.shape[:2]
    image_size = (width, height)
    if image_size != calibration_image_size:
        raise ValueError(
            f"Billedstoerrelse {image_size} matcher ikke kalibreringsstoerrelsen "
            f"{calibration_image_size}. Kalibrering og drift skal bruge samme oploesning."
        )

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        image_size,
        np.eye(3, dtype=np.float64),
        balance=balance,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3, dtype=np.float64),
        new_K,
        image_size,
        cv2.CV_32FC1,
    )
    undistorted = cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    if not np.any(undistorted):
        raise ValueError(
            "Undistort gav et helt sort billede. Kalibreringsparametrene er sandsynligvis ustabile "
            "eller passer ikke til dette billede."
        )

    return undistorted
