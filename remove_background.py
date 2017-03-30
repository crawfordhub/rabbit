import cv2
import numpy as np

#image has to be resized using the other code
# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'

#this is the input file This could be in a loop
img = cv2.imread('47.jpg')
converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,13,100])
upper_red = np.array([10,225,240])
mask0 = cv2.inRange(converted, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([135,13,100])
upper_red = np.array([180,225,240])
mask1 = cv2.inRange(converted, lower_red, upper_red)

# join my masks
skinMask = mask0+mask1

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
skinMask = cv2.dilate(skinMask, kernel, iterations = 10)
skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
skin = cv2.bitwise_and(img, img, mask = skinMask)
#this is the outputfile
cv2.imwrite('output.jpg',skin)
