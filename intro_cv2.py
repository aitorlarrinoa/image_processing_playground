"""
We will learn how to read and show an image in jpg format making use of cv2 library.
"""

# 1. Import necessary libraries
import cv2 

# 2. Define global variables
INPUT_IMAGE_PATH = "/Users/aitor/Desktop/Personal/repos/image_processing_playground/data/banana.jpg"

# 3. Code
# We load the image making use of cv2.imread()
image = cv2.imread(INPUT_IMAGE_PATH)
# We show the image making use of cv2.imshow() 
cv2.imshow("banana", image)

# We will wait undefinetely until the user press any key in order to see the image until the user wants:
cv2.waitKey(0)
# This line will destroy all emerging windows that cv2 created.
cv2.destroyAllWindows()