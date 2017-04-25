import cv2
import numpy as np
import os
import errno
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from peakdetect import peakdetect
import pymorph

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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    skinMask = cv2.dilate(skinMask, kernel, iterations = 10)
    skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
    kernel1 = np.ones((250,250),np.uint8)
    #Added these two parts
    closing = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel1)
    skin = cv2.bitwise_and(img, img, mask = closing)
    #skin = cv2.bitwise_and(img, img, mask = skinMask)
    #this is the outputfile
    print output_dir+inp
    cv2.imwrite(output_dir+inp,skin)
    return skin


#def gen_mask_2_rm_glare(file):
#read file
#img = cv2.imread('13.jpg')
#Get the green channel of the image only
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img[:,:,0]=0
#img[:,:,2]=0
#proc_cluster=img[:,:,1]
#image = proc_cluster.reshape((proc_cluster.shape[0] * proc_cluster.shape[1], 1))
#clt = KMeans(n_clusters = 7)
#clt.fit(image)
#img[img<int(max(clt.cluster_centers_)[0])]=0
#lower_red = np.array([0,0,180])
#lower_red = np.array([-1,-1,int(max(clt.cluster_centers_)[0])])
#upper_red = np.array([257,257,270])
#mask=np.zeros((img.shape[0],img.shape[1]),dtype=np.int32)
#mask=np.copy(img[:,:,1])
#mask[mask<int(max(clt.cluster_centers_)[0])]=0
#mask[mask>int(max(clt.cluster_centers_)[0])]=255
#mask = cv2.inRange(img, lower_red, upper_red)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#mask = cv2.dilate(mask, kernel, iterations = 5)
#mask = cv2.GaussianBlur(mask, (5, 5), 0)
#kernel1 = np.ones((5,5),np.uint8)
#closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
#skin = cv2.bitwise_and(img, img, mask = mask)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#read the array skin-- just the green channel
#skin_g=skin[:,:,1]
#hist, bin_edges = np.histogram(np.ndarray.flatten(skin_g), bins=255)
#cv2.imwrite('maskout.jpg',skin)


#def negative(file):
#img = cv2.imread('Test.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img[:,:,0]=0
#img[:,:,2]=0
#plt.figure(figsize=(10,8))
#plt.subplot(311)
#plot in the first cell
#plt.subplots_adjust(hspace=.5)
#plt.title("Hue")
#plt.hist(np.ndarray.flatten(img[:,:,1]), bins=255)
#hist, bin_edges = np.histogram(np.ndarray.flatten(img[:,:,1]), bins=255)
#hist=(img.shape[0]*img.shape[1])-hist
#plt.bar(bin_edges[:-1], hist, width = 1)
#plt.xlim(min(bin_edges), max(bin_edges))
#plt.show()
#P=np.histogram(np.ndarray.flatten(gray_image), bins=255)
#P.shape
#plt.show()

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




def rm_glare(filename):
    img = cv2.imread("Test5.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[:,:,0]=0
    img[:,:,2]=0
    hist=cv2.calcHist([img[:,:,1]],[0],None,[256],[0,256])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    closing = cv2.morphologyEx(hist, cv2.MORPH_CLOSE, kernel1)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel1)
    new_hist=cv2.GaussianBlur(opening,(5,5),0)
    G=np.gradient(hist_new[:,0])
    peaks=peakdetect(-1*G_new, lookahead=15)
    #choosing the lowest or highest peak b/c this is gradient
    peak=(peaks[0][-1][0]) if (peaks[0][-1][0]>peaks[1][-1][0]) else (peaks[1][-1][0])
    img[img[:,:,1]<peak]=0
    cv2.imwrite('test.jpg',img)
    #plt.imshow(img)
    #plt.show()
    #hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #hue, sat, val = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
    #plt.figure(figsize=(10,8))
    #plt.subplot(311)                             #plot in the first cell
    #plt.subplots_adjust(hspace=.5)
    #plt.title("Hue")
    #plt.hist(np.ndarray.flatten(hue), bins=255)
    #plt.subplot(312)                             #plot in the second cell
    #plt.title("Saturation")
    #plt.hist(np.ndarray.flatten(sat), bins=255)
    #plt.subplot(313)                             #plot in the third cell
    #plt.title("Luminosity Value")
    #plt.hist(np.ndarray.flatten(val), bins=255)
    #plt.show()
    #import time
    g=img[:,:,1]
    image = g.reshape((g.shape[0] * g.shape[1], 1))
    #t0 = time.time()
    clt = KMeans(n_clusters = 6)
    #t0 = time.time()
    #clt.fit(image)
    lower_red = np.array([-1,-1,190])
    upper_red = np.array([257,257,270])
    mask0 = cv2.inRange(hsv_image, lower_red, upper_red)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.dilate(mask0, kernel, iterations = 4)
    #skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
    kernel1 = np.ones((10,10),np.uint8)
    #Added these two parts
    closing = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel1)
    skin = cv2.bitwise_and(img, img, mask = closing)
    #skin = cv2.bitwise_and(img, img, mask = mask0)
    #Making a binary images
    #cv2.imwrite('maskout.jpg',skin)
    data_dir = "/Users/ezahedin/Downloads/kaggle/train/"
    outpath="/Users/ezahedin/Desktop/Kaggle_project/outputs/"



#negative('Test.jpg')
#gen_mask_2_rm_glare('13.jpg')
#type_folders = []
#for filename in os.listdir(data_dir):
#    ''' this lists:
#    type_1
#    type_2
#    type_3
#    '''
#    type_folders.append(filename)
#
#for type_f in type_folders:
#    if not os.path.exists(outpath+type_f):
#        os.mkdir(outpath+type_f)
#    if 'DS' not in type_f:
#        for image_file in os.listdir(data_dir+type_f):
#            if 'DS' not in image_file:
#                img_to_skin(data_dir+type_f+'/'+image_file,outpath,type_f+'/'+image_file)
#
