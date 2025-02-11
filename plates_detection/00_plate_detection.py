# %%
# 1. Import necessary libraries
import cv2
import numpy as np
from scipy import ndimage

# 2. Define global variables
INPUT_IMAGE_PATH = "/Users/aitor/Desktop/Personal/repos/image_processing_playground/plates_detection/data/plate3.jpg"

# 3. Code
image = cv2.imread(INPUT_IMAGE_PATH)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshold_otsu, _ = cv2.threshold(
    src=image_gray, thresh=0, maxval=255, type=cv2.THRESH_OTSU
)
mask_otsu = np.uint8(255 * (image_gray > threshold_otsu))

# We label each component:
components = cv2.connectedComponentsWithStats(
    image=mask_otsu,
    connectivity=4,
    ltype=cv2.CV_32S,
)

num_objects = components[0]
labels = components[1]
stats = components[2]

mask_objects = list()
convex_mask_list = list()

for index in range(num_objects):
    x, y, w, h, area = stats[index]
    if area > stats[:, 4].mean() / 10:
        mask = ndimage.binary_fill_holes(labels == index)
        mask = np.uint8(255 * mask)
        mask_objects.append(mask)
        # In order to know which is the plate, we are going to compare the area of the object with the area of the convex hull
        _, contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        hull = cv2.convexHull(cnt)
        convex_points = hull[:, 0, :]
        m, n = mask.shape
        area_comparison = np.zeros((m, n))
        convex_mask = np.uint8(
            255 * cv2.fillConvexPoly(area_comparison, convex_points, 1)
        )
        convex_mask_list.append(convex_mask)

# %%
""" cv2.imshow("", mask_otsu)
cv2.waitKey(0)

cv2.destroyAllWindows() """
