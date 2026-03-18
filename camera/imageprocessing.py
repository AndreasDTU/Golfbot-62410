import cv2
import numpy as np



#Turn into colorspace of choice, idk if this should go before lens distortion or after idk https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#https://learnopencv.com/color-spaces-in-opencv-cpp-python/
def imageprocessing(img, colorspace):
    return cv2.cvtColor(img,colorspace)


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



#Make picture fit altså fix lens distortion https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html



#Pitcture from hls to 5 colors(Wall, Ball, Orange Ball, Robot, Floor)
