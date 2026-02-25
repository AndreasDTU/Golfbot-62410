
import cv2


cam = cv2.VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()
if s:    # frame captured without any errors
    cv2.namedWindow("cam-test",cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow("cam-test",img)
    cv2.waitKey(0)
    cv2.destroyWindow("cam-test")
    cv2.imwrite("filename.jpg",img) #save image

    