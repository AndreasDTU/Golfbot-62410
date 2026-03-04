
import cv2


def imagecapture(CameraID):
    cam = cv2.VideoCapture(CameraID)  # Logitech USB) 
    s, img = cam.read()
    if s == True:
        cv2.imshow("Camera", img)
        cv2.imwrite("filename.jpg", img)
    return img








