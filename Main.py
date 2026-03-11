#Main code loop for golfbot
from camera.pictocord import pictocord
from camera.image import imagecapture
from camera.imageprocessing import imageprocessing
import cv2
def main():
    #Get picture from camera
    img = imagecapture(0)
    new = imageprocessing(img, cv2.COLOR_BGR2HSV) #Change to whatever colorspace we want
    cv2.imwrite("processed.jpg", new) #Save processed image for debugging
    #coordsystem = pictocord(new, 1920, 1080) #Change to whatever resolution we want

    #Pathfinding and movement code here, using coordsystem as input

if __name__ == "__main__":
    main()
