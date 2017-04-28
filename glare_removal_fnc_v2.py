import cv2
import numpy as np
import os
import errno
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from peakdetect import peakdetect
import pymorph
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour



def thresholding(mat1,mat2,threshold):
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            if (mat1[i,j]<threshold):
                mat2[i,j]=0
    return mat2

def copy_image(mat1,mat2):
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            if (mat2[i,j]>0):
                mat1[i,j,0]=255
                mat1[i,j,1]=127
                mat1[i,j,2]=100
    return mat1

##############Here is the code that removes the glare from the pictures. The input for this code is an image#########################
def find_glare_region(filef):
    img = cv2.imread(filef)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist=cv2.calcHist([img[:,:,1]],[0],None,[256],[0,256])
    kernel1 = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(hist, cv2.MORPH_CLOSE, kernel1)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel1)
    new_hist=cv2.GaussianBlur(opening,(5,5),0)
    G=np.gradient(new_hist[:,0])
    G_new=cv2.GaussianBlur(G,(5,5),0)
    peaks=peakdetect(-1*G_new, lookahead=25)
    #choosing the lowest or highest peak b/c this is gradient
    peakmax=(peaks[0][-1][0]) if (peaks[0][-1][0]>peaks[1][-1][0]) else (peaks[1][-1][0])
    peakmin=(peaks[0][-1][0]) if (peaks[0][-1][0]<peaks[1][-1][0]) else (peaks[1][-1][0])
    peak=peakmin
    if (peak<230):
        peak=230
    firstmask=np.copy(img[:,:,1])
    firstmask[firstmask>peak]=255
    firstmask[firstmask<peak+1]=0
    ret,firstmask = cv2.threshold(firstmask,127,255,0)
    #union with the open hat part -- This is the threshold for the bright spots
    #threshold=img[:,:,1].mean()+((peakmin-img[:,:,1].mean()))*.65
    threshold=img[:,:,1].max()-(img[:,:,1].mean()+img[:,:,1].mean()*.70)
    Y=0
    secondmask=np.copy(img[:,:,1])
    #secondmask[secondmask>peak-1]=0
    cv2.imwrite('f.jpg',firstmask)
    kernel0 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (11, 11))
    #kernel4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (15,15 ))
    #kernel5 = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    interval=7
    mesh=int((peak)/interval)
    Ptot=np.copy(img[:,:,1])
    Ptot=0
    Percent=100/(interval)
    Progress=0
    print 'peak is:',peak
    print 'peakmin is:', peakmin
    print 'threshold',threshold
    print 'peak median',img[:,:,1].mean()
    for x in range(0, interval):
        Progress=Progress+Percent
        print 'Progress report:',str(Progress)+'%'
        tmpmask=np.copy(secondmask)
        tmpmask[tmpmask<Y+1]=0
        tmpmask[tmpmask>Y]=255
        Y=Y+mesh
        P0=pymorph.openrecth(tmpmask,kernel0,kernel0)
        P0=thresholding(secondmask,P0,threshold)
        #cv2.imwrite(str(x)+'P0.jpg',P0)
        P1=pymorph.openrecth(tmpmask,kernel1,kernel1)
        P1=thresholding(secondmask,P1,threshold)
        #cv2.imwrite(str(x)+'P1.jpg',P1)
        P2=pymorph.openrecth(tmpmask,kernel2,kernel2)
        P2=thresholding(secondmask,P2,threshold)
        #cv2.imwrite(str(x)+'P2.jpg',P2)
        P3=pymorph.openrecth(tmpmask,kernel3,kernel3)
        P3=thresholding(secondmask,P3,threshold)
        #cv2.imwrite(str(x)+'P3.jpg',P3)
        #P4=pymorph.openrecth(tmpmask,kernel4,kernel4)
        #P4=thresholding(secondmask,P4,threshold)
        #cv2.imwrite(str(x)+'P4.jpg',P4)
        #P5=pymorph.openrecth(tmpmask,kernel5,kernel5)
        #P5=thresholding(secondmask,P5,threshold)
        #cv2.imwrite(str(x)+'P5.jpg',P5)
        Ptot=Ptot+P0+P1+P2+P3
        Ptot[Ptot>255]=255
    cv2.imwrite('f.jpg',firstmask)
    cv2.imwrite('Ptot.jpg',Ptot)
    _, contours, _= cv2.findContours(firstmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(firstmask,contours,-1,(255,255,255),-1)
    _, contours, _= cv2.findContours(firstmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(firstmask,contours,-1,(255,255,255),5)
    finalmask=firstmask+Ptot
    finalmask[finalmask>255]=255
    #_, contours, _= cv2.findContours(finalmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(finalmask,contours,-1,(255,255,255),-1)
    #ActiveC(finalmask,img)
    cv2.imwrite('finalmask.jpg',finalmask)
    #first_dilation = cv2.dilate(firstmask,kernel,iterations = 5)
    dilation = cv2.dilate(finalmask,kernel,iterations = 3)
    dst = cv2.inpaint(img,dilation,5,cv2.INPAINT_TELEA)
    cv2.imwrite('dialation.jpg',dilation)
    cv2.imwrite('outfile.jpg',dst)
    cv2.imwrite('copied.jpg',copy_image(img,finalmask))
    return contours



































