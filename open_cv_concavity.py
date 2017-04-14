import cv2
import numpy as np
import skimage
import os 

import matplotlib.pyplot as plt
for file in os.listdir('data/Train/Type_1/'):
	# img = cv2.imread('data/Train/Type_1/396.jpg',0)
	img = cv2.imread(file,0)
	ret,thresh = cv2.threshold(img,20,255,1)
	contours,hierarchy = cv2.findContours(thresh, 1, 2)

	cnt = contours[0]
	M = cv2.moments(cnt)
	print M

	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	circ1 = plt.Circle((cx,cy), 10, color='r')

	# plot stuff
	fig, ax = plt.subplots()
	ax.imshow(img)
	ax.add_artist(circ1)
	plt.show()

	area = cv2.contourArea(cnt)
	perimeter = cv2.arcLength(cnt,True)
	hull = cv2.convexHull(cnt)
	k = cv2.isContourConvex(cnt)


	print cx, cy, area, perimeter, hull, k