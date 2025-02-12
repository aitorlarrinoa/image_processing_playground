# 1. Import necessary libraries
import cv2
import numpy as np
from scipy import ndimage

# 2. Define global variables
INPUT_IMAGE_PATH = "/Users/aitor/Desktop/Personal/repos/image_processing_playground/plates_detection/data/plate2.jpg"

# 3. Code
image = cv2.imread(INPUT_IMAGE_PATH)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(image_gray, threshold1=50, threshold2=150)

kernel = np.ones((5, 5), np.uint8)
edges = cv2.dilate(edges, kernel)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    print(cnt)

cv2.imshow("", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
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


""" threshold_otsu, _ = cv2.threshold(
    src=image_gray, thresh=0, maxval=255, type=cv2.THRESH_OTSU
)
mask_otsu = np.uint8(255 * (image_gray > threshold_otsu))

# kernel = np.ones((3, 3), np.uint8)
# mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)

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
    # cv2.imshow("", np.uint8(255 * (labels == index)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    mask = ndimage.binary_fill_holes(labels == index)
    mask = np.uint8(255 * mask)
    mask_objects.append(mask)
    # In order to know which is the plate, we are going to compare the area of the object with the area of the convex hull
    contours, _ = cv2.findContours(
        image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )
    cnt = contours[0]
    hull = cv2.convexHull(cnt)
    convex_points = hull[:, 0, :]
    m, n = mask.shape
    area_comparison = np.zeros((m, n))
    convex_mask = np.uint8(255 * cv2.fillConvexPoly(area_comparison, convex_points, 1))
    convex_mask_list.append(convex_mask)

cv2.imshow("", mask_otsu)
cv2.waitKey(0)

cv2.destroyAllWindows()
 """
