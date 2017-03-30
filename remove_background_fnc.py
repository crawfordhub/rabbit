import cv2
import numpy as np
import os

#image has to be resized using the other code
# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'

#this is the input file This could be in a loop
def img_to_skin(img):
    img = cv2.imread(img)
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
    cv2.imwrite('output'+img+'.jpg',skin)

    return skin


data_dir = "../sagebrush/kaggle/Intel_Data/train/"
type_folders = []
for filename in os.listdir(data_dir):
    ''' this lists:
    type 1
    type 2
    type 3
    '''
    type_folders.append(filename)

for type_f in type_folders:
    if 'DS' not in type_f:
        for image_file in os.listdir(data_dir+type_f):
            if 'DS' not in image_file:
                img_to_skin(data_dir+type_f+image_file)

