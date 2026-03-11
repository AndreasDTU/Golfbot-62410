import cv2
import numpy as np



#Turn into colorspace of choice, idk if this should go before lens distortion or after idk https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#https://learnopencv.com/color-spaces-in-opencv-cpp-python/
def imageprocessing(img, colorspace):
    return cv2.cvtColor(img,colorspace)


def undistort_with_calibration(img, calibration_file):
    data = np.load(calibration_file)
    camera_matrix = data["camera_matrix"]
    distortion_coeffs = data["distortion_coeffs"]
    return cv2.undistort(img, camera_matrix, distortion_coeffs)



#Make picture fit altså fix lens distortion https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html



#Pitcture from hls to 5 colors(Wall, Ball, Orange Ball, Robot, Floor)
