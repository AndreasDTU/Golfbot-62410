
import cv2


def imagecapture(CameraID):
    cam = cv2.VideoCapture(CameraID)  # Logitech USB) 
    print("Camera opened: ", cam.isOpened())
    s, img = cam.read()
    cam.release()
    return img






