import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the image
image = cv2.imread("secutt.png")

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# create a binary thresholded image
_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# create an empty list to store the contour points
contour_points = []

# loop through all contours
for contour in contours:
    # loop through all points in the contour
    for point in contour:
        # extract x and y coordinates of the point
        x, y = point[0]
        # append the coordinates as a complex number to the list
        contour_points.append(complex(x, y))

# check if the first and last points are the same for the last contour
if contour_points[0] != contour_points[-1]:
    # add the first point to the end to close the curve
    contour_points.append(contour_points[0])

# convert the list of points to a numpy array for easier manipulation
contour_points = np.array(contour_points)

# calculate the centroid of the points
centroid = np.mean(contour_points, axis=0)

# center the points around the centroid
contour_points -= centroid

# calculate the maximum absolute value of the coordinates
max_abs = np.max(np.abs(contour_points))

# normalize the points to be within [-1, 1]
contour_points /= max_abs

print(contour_points.tolist())
# plot the contour points
plt.plot(contour_points.real, contour_points.imag, 'g-')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

