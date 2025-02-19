# 1. Import necessary libraries
import cv2
import numpy as np
from scipy import ndimage
# import imutils

# 2. Define global variables
INPUT_IMAGE_PATH = "/Users/aitor/Desktop/Personal/repos/image_processing_playground/plates_detection/data/plate1.jpg"

# 3. Code
image = cv2.imread(INPUT_IMAGE_PATH)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(image_gray, threshold1=30, threshold2=200)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
screenCnt = None

for cnt in contours:
    # We calculate the perimeter and the aprox poly:
    perim = cv2.arcLength(cnt, True)
    approxpoly = cv2.approxPolyDP(cnt, 0.018 * perim, True)
    # if our approximated contour has four points, then we can assume that we have found our screen
    if len(approxpoly) == 4:
        screenCnt = approxpoly
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        break
     

cv2.imshow("", image)
cv2.waitKey(0)
cv2.destroyAllWindows()









