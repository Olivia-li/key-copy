from utils import image_args
import perception
import cv2

image = cv2.imread(image_args()["image"])
contours = perception.find_contours(image)
perception.apply_transform(image, contours)
