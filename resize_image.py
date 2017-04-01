from PIL import Image
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import PIL
from scipy.ndimage import imread
import os
#So far this code onlu resize the picture to 512,512. It has to be looped over files
#I don't know how to distinguish object from background yet
# I have to create a mask to do this


#resizing the picture
basewidth = 512
hsize = 512
img = Image.open('25.jpg')
img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
img.save('resized.jpg')

#Turning images into grayscales
img1 = Image.open('resized.jpg').convert('L')
img1.save('greyscale.png')


#Standardize the pixel values
img2 = Image.open('greyscale.png')
mean = np.mean(img2)
std = np.std(img2)
#img = img-mean
img2 = img2/std
plt.hist(img2.flatten(),bins=200)
#img[img < .5] = 0.
#img[img~]
plt.show()


#This is the function that reads all the filed and remove either all white or black pictures
#data_dir = "/Users/ezahedin/Downloads/resized_data_no_background/"
data_dir="/Users/ezahedin/Desktop/Kaggle_project/resized_data_no_background/"
#reading the folders in directory
def cleandata(dir):
    dir_names=[x[0] for x in os.walk(dir)]
    del dir_names[0]
    del dir_names[1:3]
    print dir_names
    for subdir in dir_names:
        for image_file in os.listdir(subdir):
            if 'DS' not in image_file:
                img=Image.open(subdir+"/"+image_file)
                if sum(img.convert("L").getextrema()) in (0, 2):
                    os.remove(subdir+"/"+image_file)



