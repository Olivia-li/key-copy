import numpy as np
import cv2
from skimage import filters
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image file path")
image_args = vars(ap.parse_args())

image = cv2.imread(image_args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

cv2.imshow("Original", image)
cv2.imshow("Canny Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


def order_rectangle_points(vertices):
    # order is [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")

    s = vertices.sum(axis=1)
    # top-left
    rect[0] = vertices[np.argmin(s)]
    # bottom-right
    rect[2] = vertices[np.argmax(s)]

    diff = np.diff(vertices, axis=1)
    # top-right
    rect[1] = vertices[np.argmin(diff)]
    # bottom-left
    rect[3] = vertices[np.argmax(diff)]

    return rect


def perception_transform(image, vertices):
    rect = order_rectangle_points(vertices)
    (tl, tr, br, bl) = rect

    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(height_left), int(height_right))

    destination_size = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(rect, destination_size)
    warped = cv2.warpPerspective(
        image, transform_matrix, (max_width, max_height))
    return warped
