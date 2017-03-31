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


def resize_img(fileinput,output_dir,inp):

    
    #resizing the picture
    basewidth = 512
    hsize = 512
    img = Image.open(fileinput+inp)
    img = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
    img.save(output_dir+inp)
    return

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
    # img[img < .5] = 0.
    #img[img~]
    #plt.show()
    
    return img2

data_dir = "/Users/dancrawford/sagebrush/kaggle/Intel_Data/train/"
outpath="/Users/dancrawford/rabbit/resized_train/"



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
                resize_img(data_dir,outpath,type_f+'/'+image_file)
