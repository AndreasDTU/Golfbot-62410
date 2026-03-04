#Calculate the threshhold for the image processing, using the HSV color space. This is done by taking a sample of the image and calculating the average color of the sample. The threshhold is then set to be a certain distance from the average color in the HSV color space.
#HSV is used since it is easier to separate colors in the HSV color space than in the RGB color space. The hue component of the HSV color space represents the color, while the saturation and value components represent the intensity of the color. By setting a threshhold based on the hue component, we can easily separate different colors in the image.

import cv2
import numpy as np  

def calcthreshhold(img, sample_size):
    #Take a sample of the image
    sample = img[0:sample_size, 0:sample_size]
    
    #Calculate the average color of the sample
    avg_color = cv2.mean(sample)[:3] #Get the average color in BGR format
    
    #Convert the average color to HSV format
    avg_color_hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0][0]
    
    #Set the threshhold to be a certain distance from the average color in the HSV color space
    threshhold = (avg_color_hsv[0] - 10, avg_color_hsv[1] - 50, avg_color_hsv[2] - 50), (avg_color_hsv[0] + 10, avg_color_hsv[1] + 50, avg_color_hsv[2] + 50)
    
    return threshhold