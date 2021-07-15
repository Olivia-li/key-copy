from scipy import spatial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import subplots
from numpy import load, stack
import numpy as np
from stl import mesh
import cv2

image = cv2.imread('key_flat.png', cv2.COLOR_BGR2GRAY)
stl_image = image.copy()
print(stl_image.shape)
stl_image = stl_image[:, :, np.newaxis]
print(stl_image.shape)


x = load(stl_image[0])
y = load(stl_image[1])
z = load(stl_image[2])

points = stack((x, y, z), axis=-1)

v = spatial.ConvexHull(points)
fig, ax = subplots(subplot_kw=dict(projection='3d'))
ax.plot_trisurf(*v.points.T, triangles=v.simplices.T)
fig.show()
cv2.imshow("stl", stl_image)
cv2.waitKey(0)
