'''
Created on Mar 30, 2017

@author: Jas
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

import PIL
from PIL import Image

imagePath = "1013.jpg"

im = Image.open(imagePath)
x = im.split()
print x
r,g,b = im.split()
left,upper,right,lower = im.getbbox()
tempImage = im.crop((left,upper,right,lower))
newPath = imagePath.split(".")[0]+"_Cropped" + "." + imagePath.split(".")[1]
tempImage.save(newPath)