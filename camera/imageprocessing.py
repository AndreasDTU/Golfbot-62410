import cv2
import numpy as np



#Turn into colorspace of choice, idk if this should go before lens distortion or after idk
def imageprocessing(img, colorspace):
    return cv2.cvtColor(img,colorspace)



#Make picture fit altså fix lens distortion https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html



#Pitcture from hls to 5 colors(Wall, Ball, Orange Ball, Robot, Floor)