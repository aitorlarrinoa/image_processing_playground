# 1. Import necessary libraries
import cv2
import numpy as np

# 2. Define global variables
INPUT_IMAGE_PATH = "/Users/aitor/Desktop/Personal/repos/image_processing_playground/numbers_detection/data/plate1.jpg"

# 3. Code
image = cv2.imread(INPUT_IMAGE_PATH)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshold_otsu, _ = cv2.threshold(
    src=image_gray, thresh=0, maxval=255, type=cv2.THRESH_OTSU
)
mask_otsu = np.uint8(255 * (image_gray < threshold_otsu))

# We label each component:
components = cv2.connectedComponentsWithStats(
    image=mask_otsu,
    connectivity=4,
    ltype=cv2.CV_32S,
)

num_objects = components[0]
labels = components[1]
stats = components[2]

for index in range(num_objects):
    x, y, w, h, area = stats[index]


cv2.imshow("", mask_otsu)
cv2.waitKey(0)

cv2.destroyAllWindows()
