import cv2
import numpy as np



#Turn into colorspace of choice, idk if this should go before lens distortion or after idk https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#https://learnopencv.com/color-spaces-in-opencv-cpp-python/
def imageprocessing(img, colorspace):
    return cv2.cvtColor(img,colorspace)


def undistort_with_calibration(img, calibration_file, balance=0.5):
    data = np.load(calibration_file)
    K = data["K"]
    D = data["D"]

    height, width = img.shape[:2]
    image_size = (width, height)
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
        cv2.CV_16SC2,
    )
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)



#Make picture fit altså fix lens distortion https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html



#Pitcture from hls to 5 colors(Wall, Ball, Orange Ball, Robot, Floor)
