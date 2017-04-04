'''
Created on Mar 30, 2017

@author: Jas
'''

from PIL import Image

imagePath = "1013.jpg"

im = Image.open(imagePath)
left,upper,right,lower = im.getbbox()
tempImage = im.crop((left,upper,right,lower))
newPath = imagePath.split(".")[0]+"_Cropped" + "." + imagePath.split(".")[1]
tempImage.save(newPath)
