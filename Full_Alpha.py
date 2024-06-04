import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

import cv2

# read the image
image = cv2.imread("Test_001.png")

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


#Fonctions____________________________________________________________________________________________
def add_points_between_segments(points, n):
    new_points = []
    for i in range(len(points) - 1):
        for j in range(n + 1):
            # Calculer les coordonnées du point intermédiaire
            x = points[i].real + j * (points[i + 1].real - points[i].real) / (n + 1)
            y = points[i].imag + j * (points[i + 1].imag - points[i].imag) / (n + 1)
            new_points.append(complex(x, y))
    new_points.append(points[-1])
    return new_points

#Hyperparamètres
M = 51  #Précision des cercles
N = 200  # Nombre de points
nb_interpol = 150

#Transformée de fourrier______________________________________________________________________________


# Fonction Z(t) représentant une courbe fermée dans le plan complexe
def Z1(t):
    return np.exp(2j * np.pi * t) + 0.5 * np.exp(4j * np.pi * t)

fleche = [-2j, 2+2j, 1j, -2+2j, -2j]
ruban= [1+5j, 1j, -1+5j, -1-2j, 1-2j, 1+2j]
sqooch = [1.7+3.3j, 1.4+2.3j, 1.7+1.7j, 2.4+1.4j, 3.2+1.7j,4.2+1.9j,5.1+2.2j,
          6+2.4j, 6.9+2.7j, 7.8+2.9j, 9.2+3.2j, 7.5+2.5j, 6.4+2j,
          5.1+1.5j, 4+1j, 2.7+0.5j, 1.6+0.2j, 0.7+0.5j, 0.5+1.2j,
          0.7+1.9j, 1.2+2.6j, 1.7+3.3j]
pluscross_shape = [3+1j, 1+1j, 1+3j, -1+3j, -1+1j, -3+1j, -3-1j, -1-1j, -1-3j, 1-3j, 1-1j, 3-1j, 3+1j]
square_points = contour_points#[1+1j, -1+1j, -1-1j, 1-1j, 1+1j]

# Fonction Z(t) parcourant la liste de points en fonction du temps

def Z(x):
    return interpolated_points(x)

# Paramètres d'échantillonnage temporel

t = np.linspace(0.0, 1.0, N, endpoint=True)  # Valeurs de temps

#z1 = interp1d(np.linspace(0, 1, len(square_points)), square_points, kind='linear')(np.linspace(0.0, 1.0, 25))
#z2 = interp1d(np.linspace(0, 1, len(z1)), z1, kind='cubic', fill_value='extrapolate')(t)

interpolated_points = interp1d(np.linspace(0, 1, len(square_points)), square_points, kind='zero')(t)

new_inter = add_points_between_segments(square_points, nb_interpol)


plt.plot(np.real(new_inter), np.imag(new_inter))

# Calcul de Z(t)
z = new_inter

# Calcul des coefficients de Fourier complexes en tronquant la transformée de Fourier
# Nombre de coefficients à calculer
coefficients = np.fft.fft(z)/ N

organized_coefficients = [coefficients[0]]
for i in range(M-1):
    if i%2 == 0:
        organized_coefficients.append(coefficients[-(i//2+1)])
    else:
        organized_coefficients.append(coefficients[(i//2+1)])

indices = [0] + [i for j in range(1, M//2+1) for i in (-j, j)]
#indices = indices[2:]
#print(indices)

new_coeffficients = [(indices[i], abs(organized_coefficients[i])) for i in range(M)]

# Filtrer les coefficients dont le module est inférieur à 1e-12
coefficients_filtres = [(n, abs(c), np.angle(c)) for n, c in enumerate(coefficients)if abs(c) > 1e-13]
#print(coefficients_filtres)




#Dessin des cercles___________________________________________________________________________________

circles = new_coeffficients[1:]
nb_points = 500

theta = np.linspace(0, 4*np.pi, nb_points)


all_x = [circles[i][1]*np.cos(theta*circles[i][0]) for i in range(len(circles))]
all_y = [circles[i][1]*np.sin(theta*circles[i][0]) for i in range(len(circles))]



cord_circles_x = [sum(all_x[:i]) for i in range(1, len(all_x)+1)]
cord_circles_y = [sum(all_y[:i]) for i in range(1, len(all_y)+1)]

bordures = [min(np.array(cord_circles_x).flatten()), max(np.array(cord_circles_x).flatten()),
min(np.array(cord_circles_y).flatten()), max(np.array(cord_circles_y).flatten())]

x = cord_circles_x[-1]
y = cord_circles_y[-1]


draw_circle = [plt.Circle((0, 0), circles[i][1], color='b', fill=False) for i in range(len(circles))]



def init():
    for circle2draw in draw_circle:
        ax.add_artist(circle2draw)
    ax.set_aspect('equal')
    return draw_circle,


def animate(t):
    line.set_data(x[:t], y[:t])
    for i in range(1, len(draw_circle)):
        draw_circle[i].center = (cord_circles_x[i-1][t], cord_circles_y[i-1][t])
    scat.set_offsets(np.column_stack(([c.center[0] for c in draw_circle], [c.center[1] for c in draw_circle]))) 
    last_scat.set_offsets(np.column_stack((cord_circles_x[-1][t], cord_circles_y[-1][t])))
    return line,draw_circle, scat, last_scat

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_xlim(-1.2*max(bordures), 1.2*max(bordures))
ax.set_ylim(-1.2*max(bordures), 1.2*max(bordures))

line, = ax.plot([], [], 'g-', lw=2)

scat = ax.scatter([c.center[0] for c in draw_circle], [c.center[1] for c in draw_circle], color="r", s=20)
last_scat = ax.scatter(0, 0, color='purple', marker='D', s=10)

ani = FuncAnimation(fig, animate, frames=range(nb_points), init_func=init,
                    blit=True, interval=50)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Animation du tracé de deux cercles')
#plt.grid(True)
plt.show()
