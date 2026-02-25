
import cv2


cam = cv2.VideoCapture(1)  # Logitech USB) 
image = cv2.imread("Bane.jpg")
while True:  
    s, img = cam.read()
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)
    cv2.waitKey(1)    # wait for 1 millisecond before moving on to next frame
    cv2.imwrite("filename.jpg", hsv)

