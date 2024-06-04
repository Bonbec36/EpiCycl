import cv2
import numpy as np
import matplotlib.pyplot as plt

def separate_contours(image):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Créer une image binaire avec une inversion des couleurs
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

    # Trouver les contours avec séparation des contours intérieurs et extérieurs
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Séparer les contours intérieurs et extérieurs
    outer_contours = []
    inner_contours = []

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # Contours extérieurs
            outer_contours.append(contours[i])
        else:  # Contours intérieurs
            inner_contours.append(contours[i])

    return outer_contours, inner_contours

# Charger l'image
img_src = 'images/secutt.png'
img = cv2.imread(img_src)
# Convertir en RGB
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Séparer les contours
outer_contours, inner_contours = separate_contours(image)

# create an empty list to store the contour points
contour_points = []

print(len(outer_contours))
for contour in outer_contours:
    if len(contour) > 100:
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


# plot the contour points
plt.plot(contour_points.real, -contour_points.imag, 'g-')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()