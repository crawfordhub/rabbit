"""
Otsu thresholding
==================

This example illustrates automatic Otsu thresholding.
"""
from scipy.ndimage import imread
import matplotlib.pyplot as plt
from skimage import data
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from skimage import exposure

camera = data.camera()
#camera = io.imread('greyscale.png')
camera= imread('greyscale.png')
val = filters.threshold_otsu(camera)

hist, bins_center = exposure.histogram(camera)

plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.imshow(camera, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(132)
plt.imshow(camera < val, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(133)
plt.plot(bins_center, hist, lw=2)
plt.axvline(val, color='k', ls='--')

plt.tight_layout()
plt.show()
