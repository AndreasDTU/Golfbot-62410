
import cv2

x_max = 1920
y_max = 1080



cam = cv2.VideoCapture(1)  # Logitech USB) 
image = cv2.imread("Bane.jpg")
while True:  
    s, img = cam.read()
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)
    cv2.waitKey(1)    # wait for 1 millisecond before moving on to next frame
    cv2.imwrite("filename.jpg", hsv)

#Make 2d array for coordinate system


#Some shit like this idk???
coordinate_array = [[0 for _ in range(x_max)] for _ in range(y_max)]  # Initialize a 2D array for 640x480 image coordinates

#Make picture fit altså fix lens distortion

#Pitcture from hls to 5 colors(Wall, Ball, Orange Ball, Robot, Floor)
#Color gets an id
#Fill 2d array with color values



