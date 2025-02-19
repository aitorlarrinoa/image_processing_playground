# 1. Import necessary libraries
import cv2
import numpy as np
from scipy import ndimage

# 2. Define global variables
INPUT_IMAGE_PATH = "/Users/aitor/Desktop/Personal/repos/image_processing_playground/plates_detection/data/plate3.jpg"

# 3. Code
image = cv2.imread(INPUT_IMAGE_PATH)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Let's apply gaussian blur in order to get rid of noise in the image:
image_gray = cv2.GaussianBlur(src=image_gray,ksize=(3,3),sigmaX=0)

thresh_otsu, _ = cv2.threshold(src=image_gray, thresh=0, maxval=255, type=cv2.THRESH_OTSU)
mask_otsu = np.uint8(255*(image_gray>thresh_otsu))
# image_gray = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                     cv2.THRESH_BINARY_INV, 11, 2)  # Adjust block size (11) and C (2)

components = cv2.connectedComponentsWithStats(
    image=mask_otsu,
    connectivity=4,
    ltype=cv2.CV_32S,
)

num_objects = components[0]
labels = components[1]
stats = components[2]

mask_list = list()
convex_mask_list = list()
diffs_area = list()


# We skip the first object because it refers to the backgroud.
for index in range(1, num_objects):
    x, y, w, h, area = stats[index]
    # First, we need to fill holes of the mask:
    #mask_filled = ndimage.binary_fill_holes(labels == index)
    mask = np.uint8(255*(labels==index))
    mask_list.append(mask)

    # Convex hull calculation:
    contours, _ = cv2.findContours(
        image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )
    cnt = contours[0]
    # x, y, w, h = cv2.boundingRect(cnt)
    # area_rect = h*w
    hull = cv2.convexHull(cnt)
    convex_points = hull[:, 0, :]
    m, n = mask.shape

    area_comparison = np.zeros((m, n))
    convex_mask = np.uint8(255 * cv2.fillConvexPoly(area_comparison, convex_points, 1))
    convex_mask_list.append(convex_mask)

    # if w/h > 1:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    thing_to_show = np.uint8(image_gray*mask)
    thing_to_show_convex = np.uint8(image_gray*convex_mask)
    cv2.imshow("", thing_to_show)
    cv2.waitKey(0)

    cv2.imshow("", thing_to_show_convex)
    cv2.waitKey(0)

    area_obj = np.sum(mask)/255
    area_convex = np.sum(convex_mask)/255

    diffs_area.append(abs(area_convex-area_obj))

# cv2.imshow("", image)
# cv2.waitKey(0)
# plate_mask = mask_list[np.argmin(diffs_area)]
# print(plate_mask)

#cv2.imshow("", plate_mask)
#cv2.waitKey(0)

#cv2.imshow("", mask_otsu)
#cv2.waitKey(0)
cv2.destroyAllWindows()

""" image = cv2.imread(INPUT_IMAGE_PATH)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(image_gray, threshold1=50, threshold2=150)

kernel = np.ones((5, 5), np.uint8)
edges = cv2.dilate(edges, kernel)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    print(cnt)

cv2.imshow("", edges)
cv2.waitKey(0)
cv2.destroyAllWindows() """
""" contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

plate_rect = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    x, y, w, h = cv2.boundingRect(approx)

    aspect_ratio = w / float(h)
    if 2 < aspect_ratio < 5:  # Relación de aspecto típica de una matrícula
        plate_rect = (x, y, w, h)
        break  # Tomamos el primer rectángulo válido


if plate_rect is not None:
    x, y, w, h = plate_rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 10)  # Rectángulo verde

    cv2.imshow("Detected Plate", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se encontró la matrícula.") """
