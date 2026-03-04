
import cv2


def imagecapture(CameraID):
    cam = cv2.VideoCapture(CameraID)  # Logitech USB) 
    s, img = cam.read()
    if s == True:
        cv2.imwrite("filename.jpg", img)
    cam.release()
    return img
imagecapture(1)






