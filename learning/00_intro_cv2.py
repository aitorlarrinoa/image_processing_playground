"""
We will learn how to read and show an image in jpg format making use of cv2 library. Furthermore, we will
also learn how to show the histogram of an image.
"""

#### 1. Import necessary libraries ####
import cv2 
import matplotlib.pyplot as plt

#### 2. Define global variables ####
INPUT_IMAGE_PATH = "/Users/aitor/Desktop/Personal/repos/image_processing_playground/data/car.jpg"

#### 3. Code ####

# We load the image making use of cv2.imread()
image = cv2.imread(INPUT_IMAGE_PATH)

# Now we will calculate the histogram of the image. First of all we need to transform the image into gray 
# scale, in order to have one value per pixel. # It is quite interesting to convert the image into white-black image. 
# This will change the RGB situation, having just one value for each pixel instead of three (as in RGB). Since we are 
# dealing with 8 bits, we have 256 possible values for the pixel. Thus, 0 is considered to be black and 255 is white 
# whereas intermediate values will be gray. We use cv2.cvtColor in order to perform it:
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# We show the image making use of cv2.imshow() 
cv2.imshow("car_gray", image_gray)

# We need to flatten the array of image_gray now. This is done because in order to plot the histogram, we just
# want to focus on the values of the array, not in the positioning of each of them:
image_gray = image_gray.flatten()

# Now the histogram can be shown:
plt.figure()
plt.hist(image_gray, bins=100)
plt.show()

# We will wait undefinetely until the user press any key in order to see the image until the user wants:
cv2.waitKey(0)
# This line will destroy all emerging windows that cv2 created.
cv2.destroyAllWindows()