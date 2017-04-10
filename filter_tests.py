# DOING SCIKIT-IMAGE Tutorial.........could be useful to someone 

import numpy as np
from skimage import data
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from skimage.feature import canny


def save_img(image_array, image_name):
	Image.fromarray(image_array.astype(float)).save(image_name+".tiff")
	return

# data

coins1 = data.coins()

print coins1.shape
print type(coins1)

coins = Image.open("/Users/dancrawford/rabbit/data/train/Type_1/0.jpg")#data.coins()
cx, cy = coins.size
coins = np.array(coins)

print coins.shape
print type(coins1)


print cx, cy, cx*cy, 3*cx*cy
coins.reshape((3, cx,cy))

# histogram
hist, bin_edges = np.histogram(coins, bins=np.arange(0, 256))

img1 = Image.fromarray(coins, "RGB")
img1.save("coins_bare.tiff")
plt.bar(bin_edges[:-1], hist, width = 1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.savefig("histo.tiff")   

# Canny Stuff
edges = 1*canny(color.rgb2gray(coins)/255.)
img2 = Image.fromarray(edges.astype(float))
img2.save("canny_coins.tiff")

# ND Images
import scipy.ndimage
fill_coins = scipy.ndimage.binary_fill_holes(edges)
img3 = Image.fromarray(fill_coins.astype(float))
img3.save("fill_coins.tiff")

# Removing small shapes

label_objects, nb_labels = scipy.ndimage.label(fill_coins)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
coins_cleaned = mask_sizes[label_objects]
Image.fromarray(coins_cleaned.astype(float)).save("coins_cleaned.tiff")

# Region-based segmentation
markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2
# save_img(markers, "markers")

# Sobel
from skimage.filters import sobel
elevation_map = sobel(coins)
Image.fromarray(elevation_map).save("elevation_map.tiff")

# Morphology -> watershed
from skimage.morphology import watershed
segmentation = watershed(elevation_map, markers)
segmentation = scipy.ndimage.binary_fill_holes(segmentation - 1)
save_img(segmentation, "segmentation")

# Labeling image
from skimage.color import label2rgb

label_coins, _ = scipy.ndimage.label(segmentation)
image_label_overlay = label2rgb(label_coins, image=coins)
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].imshow(coins, cmap=plt.cm.gray, interpolation='nearest')
axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
axes[1].imshow(image_label_overlay, interpolation='nearest')

for a in axes:
    a.axis('off')
    a.set_adjustable('box-forced')

plt.tight_layout()
plt.savefig("image_label_overlay.tiff")