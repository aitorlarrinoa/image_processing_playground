""" 
We will learn how to use the OTSU method in order to find borders over a figure in an image.
"""
#### 1. Import necessary libraries ####
import cv2 
import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

#### 2. Define global variables ####
INPUT_IMAGE_PATH = "/Users/aitor/Desktop/Personal/repos/image_processing_playground/data/car.jpg"

#### 3. Code ####
# First of all, we load the image
image = cv2.imread(INPUT_IMAGE_PATH)

# As we saw in the introduction to cv2, we can convert an image into gray scale and we can plot an histogram.
# In fact, this histogram gives us information about in what values of pixels (more or less) the object is located.
# Let's take what we did in the intro_cv2.py script:
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray_flatten = image_gray.flatten()

""" plt.figure()
plt.hist(image_gray_flatten, bins=100)
plt.show() """

# Looking at the histogram, we can set a threshold of 251 in order to find the object, making use of a binary
# image:
thres = 251
# Our binary mask is going to be all white except from those pixels that are below the threshold:
mask = np.uint8((image_gray<thres)*255) 

""" cv2.imshow("mask", mask)
cv2.waitKey(0) """

# So, we can detect quite well the image making use of a chosen threshold. However, choosing the threshold by hand
# is not easy in some images. Thus, the idea is to use Otsu's method in order to look for it. The idea here is
# to find the threshold that discriminates between the background and the object looking at the hstogram.
# This can be find by using cv2.threshold() and the method cv2.THRESH_OTSU

threshold_otsu, _ = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU)
print(threshold_otsu)
mask_otsu = np.uint8((image_gray<threshold_otsu)*255) 

""" cv2.imshow("mask_otsu", mask_otsu)
cv2.waitKey(0) """

# As a bonus, if an image has more than one object and we want to label them and select the object of interest, we can do the 
# following:
output = cv2.connectedComponentsWithStats(mask_otsu, 4, cv2.CV_32S)
# The number of objects will be in position 0, labels in position 1 and stats in position 2:
num_objects = output[0]
labels = output[1]
stats = output[2]
# This can be useful when we have just one objectand the Otsu method detects some noise and we want to delete it. In that case
# we consider the object with highest area and the rest of the objects we set them to black. Ifwe want to do this now:

# stats' last column indicates the number of pixels each object has. Thus, if we consider the second object with the most pixels, 
# we can take the object with highest area. Why the second? Because the first one is the background.
# First row, row 0 is the background, we avoid it. But, we need to find where the object is by using np.argmax.
print(stats)
print(labels)
new_mask = np.uint8(labels == np.argmax(stats[:, 4][1:])+1)

# We will use ndimage in order to fill holes in case our mask has holes:
#new_mask = ndimage.binary_fill_holes(new_mask).astype(np.uint8)
cv2.imshow("mask_highest_area", new_mask)
cv2.waitKey(0)

# Finally we destroy all windows:
cv2.destroyAllWindows() 
