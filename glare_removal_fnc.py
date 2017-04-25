import cv2
import numpy as np
import os
import errno
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from peakdetect import peakdetect
import pymorph



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
    if (peak<200):
        peak=200
    firstmask=np.copy(img[:,:,1])
    firstmask[firstmask>peak]=255
    firstmask[firstmask<peak+1]=0
    ret,firstmask = cv2.threshold(firstmask,127,255,0)
    #union with the open hat part -- This is the threshold for the bright spots
    Y=255-int((img[:,:,1].max()-img[:,:,1].mean())*1.3)
    Y=int(img[:,:,1].mean()/3)
    secondmask=np.copy(img[:,:,1])
    
    kernel0 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (13, 13))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (17,17 ))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_CROSS, (19, 19))
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    interval=10
    mesh=int((peak-Y)/interval)
    Ptot=np.copy(img[:,:,1])
    Ptot=0
    Percent=100/(mesh-3)
    Progress=0
    for x in range(1, mesh-3):
        Progress=Progress+Percent
        print 'Progress report:',str(Progress)+'%'
        tmpmask=np.copy(secondmask)
        tmpmask[tmpmask<Y+1]=0
        tmpmask[tmpmask>Y]=255
        Y=Y+interval
        P0=pymorph.openrecth(tmpmask,kernel0,kernel0)
        P1=pymorph.openrecth(tmpmask,kernel1,kernel1)
        P2=pymorph.openrecth(tmpmask,kernel2,kernel2)
        P3=pymorph.openrecth(tmpmask,kernel3,kernel3)
        P4=pymorph.openrecth(tmpmask,kernel4,kernel4)
        P5=pymorph.openrecth(tmpmask,kernel5,kernel5)
        Ptot=Ptot+P0+P1+P2+P3+P4+P5
        Ptot[Ptot>255]=255
    finalmask= firstmask+Ptot
    finalmask[finalmask>255]=255
    _, contours, _= cv2.findContours(finalmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(finalmask,contours,-1,(255,255,255),-1)
    print 'Progress report: 100%'
    dilation = cv2.dilate(finalmask,kernel,iterations = 1)
    dst = cv2.inpaint(img,dilation,3,cv2.INPAINT_NS)
    cv2.imwrite('outfile.jpg',dst)


