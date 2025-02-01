""" 
We will try to find the mask of a picture of myself. The idea is to use the Canny algorithm in order to find 
borders and then try to find the object.
"""

#### 1. Import necessary libraries ####
import cv2 
import numpy as np

#### 2. Define global variables ####
INPUT_IMAGE_PATH = "/Users/aitor/Desktop/Personal/repos/image_processing_playground/data/myself.jpg"

#### 3. Code ####
image = cv2.imread(INPUT_IMAGE_PATH)

# Convert it to grayscale:
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshold_otsu, _ = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU)
mask_otsu = np.uint8((image_gray<threshold_otsu)*255) 

#cv2.imshow("otsu_mask", mask_otsu)
#cv2.waitKey(0)

# The OTSU threshold does not work correctly in finding myself in the picture. Let's go further then.
# We will use Canny in order to detect borders. Instead of finding the mask itself, we are going to detect borders first:
borders = cv2.Canny(image_gray, 25, 150)

#cv2.imshow("borders", borders)
#cv2.waitKey(0)

# As we can see, borders have been detected quite well in the image. However, it looks like we have very thin borders. This is 
# due to the Canny algorithm because it just leaves one pixel in the border. Thus, we can pass a kernel in order to make it thicker.
kernel = np.ones((5, 5), np.uint8)
borders_new = cv2.dilate(borders, kernel)
#cv2.imshow("borders_new", borders_new)
#cv2.waitKey(0)

# Once borders are detected, we want to find the mask. This is, myself. Thus, we will find contourss:
contours, _ = cv2.findContours(borders_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
objects = borders_new.copy()

cv2.drawContours(objects, [max(contours, key=cv2.contourArea)], -1, 255, thickness=-1)
#cv2.imshow("objects", objects)
#cv2.waitKey(0)
# We can observe that what we are in fact detecting is my face. 
# Now let's segment the obtained mask:
objects = objects/255
seg = np.zeros(image.shape)

# All channels: red, green and blue:
seg[:,:,0] = objects*image[:, :, 0]+255*(objects==0)
seg[:,:,1] = objects*image[:, :, 1]+255*(objects==0)
seg[:,:,2] = objects*image[:, :, 2]+255*(objects==0)

seg = np.uint8(seg)

cv2.imshow("segment", seg)
cv2.waitKey(0)
# Well it is not the finnest result we could get, but I think this is a little step. We will learn how to fix 
# the image in order to be smoother in furthers .pys.

# Finally we destroy all windows:
cv2.destroyAllWindows() 