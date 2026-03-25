import cv2

img = cv2.imread("Bane.jpg")
cv2.imwrite("Bane_hsv.jpg", cv2.cvtColor(img,cv2.COLOR_BGR2HSV))