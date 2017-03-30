import cv2
import numpy as np
import os
import errno

#image has to be resized using the other code
# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'

#this is the input file This could be in a loop
def img_to_skin(fileinput,output_dir,inp):
    img = cv2.imread(fileinput)
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,25,75])
    upper_red = np.array([10,225,255])
    mask0 = cv2.inRange(converted, lower_red, upper_red)
    
    # upper mask (170-180)
    lower_red = np.array([120,25,75])
    upper_red = np.array([180,225,255])
    mask1 = cv2.inRange(converted, lower_red, upper_red)

    # join my masks
    skinMask = mask0+mask1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.dilate(skinMask, kernel, iterations = 10)
    skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
    skin = cv2.bitwise_and(img, img, mask = skinMask)
    #this is the outputfile
    print output_dir+inp
    cv2.imwrite(output_dir+inp,skin)
    return skin


data_dir = "/Users/ezahedin/Downloads/kaggle/train/"
outpath="/Users/ezahedin/Desktop/Kaggle_project/outputs/"



type_folders = []
for filename in os.listdir(data_dir):
    ''' this lists:
    type_1
    type_2
    type_3
    '''
    type_folders.append(filename)

for type_f in type_folders:
    if not os.path.exists(outpath+type_f):
        os.mkdir(outpath+type_f)
    if 'DS' not in type_f:
        for image_file in os.listdir(data_dir+type_f):
            if 'DS' not in image_file:
                img_to_skin(data_dir+type_f+'/'+image_file,outpath,type_f+'/'+image_file)

