from utils import image_args
import perception
import measurement
from skimage import filters
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

image = cv2.imread(image_args()["image"])
orig_contours = perception.find_contours(image)
file_name = perception.apply_transform(image, orig_contours)
measurement.grab_key(file_name)
