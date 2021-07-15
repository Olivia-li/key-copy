from utils import image_args
import perception
from skimage import filters
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


def grab_key(file_path):
    flat_image = cv2.imread(file_path)
    gray = cv2.cvtColor(flat_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 75, 200)

    # Close gap between canny edge detection
    edged = cv2.dilate(gray, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    edged = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)[1]

    # find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    for c in cnts:
        if cv2.contourArea(c) > 15000 and cv2.contourArea(c) > 100000:
            raise "Could not find key bounding box"
        orig = flat_image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        # box contour around key
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # contour of the key
        cv2.drawContours(orig, cnts, -1, (0, 255, 0), 3)
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        cv2.imshow("image", imutils.resize(orig, height=650))
        cv2.waitKey(0)
