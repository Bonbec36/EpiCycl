import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

resol = 100

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

def smooth_curve(points, resolution=resol):
    # Convert the list of points to a numpy array
    points = np.array(points)

    # Compute the distance between consecutive points
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    d = np.sqrt(dx**2 + dy**2)

    # Compute the total length of the curve
    length = np.sum(d)

    # Compute the number of segments to use
    segments = int(length * resolution)

    # Create an array to store the smoothed curve
    smoothed_curve = np.zeros((segments, 2))

    # Compute the coordinates of the smoothed curve
    index = 0
    for i in range(len(points)-1):
        x0, y0 = points[i]
        x1, y1 = points[i+1]
        length = d[i]
        nx = int(length * resolution)
        xs = np.linspace(x0, x1, nx+1)[:-1]
        ys = np.linspace(y0, y1, nx+1)[:-1]
        smoothed_curve[index:index+nx, 0] = xs
        smoothed_curve[index:index+nx, 1] = ys
        index += nx

    return smoothed_curve


def smooth_right_angles(points, radius):
    smoothed_points = []
    n = len(points)
    
    for i in range(n):
        prev_point = points[i-1]
        current_point = points[i]
        next_point = points[(i+1) % n]
        
        # Vecteurs entre les points
        v1 = current_point - prev_point
        v2 = next_point - current_point
        
        # Produit scalaire pour calculer l'angle
        dot_product = np.real(v1) * np.real(v2) + np.imag(v1) * np.imag(v2)
        angle = np.arccos(dot_product / (np.abs(v1) * np.abs(v2)))
        
        # Vérification de l'angle droit
        if np.abs(angle - np.pi / 2) < 1e-5:
            # Points d'arc de cercle
            num_points = int(np.ceil(np.abs(np.angle(v1) - np.angle(v2)) * radius))
            theta = np.linspace(np.angle(v1), np.angle(v2), num_points)
            arc_points = current_point + radius * np.exp(1j * theta)
            
            # Ajouter les points d'arc de cercle
            smoothed_points.extend(arc_points[:-1])  # Exclure le dernier point pour éviter les doublons
        else:
            # Ajouter le point tel quel
            smoothed_points.append(current_point)
    
    return smoothed_points



#Hyperparamètres
M = 51  #Précision des cercles
N = 100  # Nombre de points
nb_interpol = 100


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
corner_shape = [1j, -1+1j, -1-1j, -1j, 0, 1, 1+1j,1j]
square_points = [1+1j, -1+1j, -1-1j, 1-1j, 1+1j]

# Fonction Z(t) parcourant la liste de points en fonction du temps

def Z(x):
    return interpolated_points(x)

# Paramètres d'échantillonnage temporel

t = np.linspace(0.0, 1.0, N, endpoint=True)  # Valeurs de temps

#z1 = interp1d(np.linspace(0, 1, len(square_points)), square_points, kind='linear')(np.linspace(0.0, 1.0, 25))
#z2 = interp1d(np.linspace(0, 1, len(z1)), z1, kind='cubic', fill_value='extrapolate')(t)

interpolated_points = interp1d(np.linspace(0, 1, len(square_points)), square_points, kind='zero')(t)

new_inter = add_points_between_segments(square_points, nb_interpol)

new_inter_rounded = smooth_right_angles(new_inter, radius=0.1)


plt.plot(np.real(new_inter_rounded), np.imag(new_inter_rounded))



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

new_coeffficients = [(indices[i], abs(organized_coefficients[i]), np.angle(organized_coefficients[i])) for i in range(M)]

# Filtrer les coefficients dont le module est inférieur à 1e-12
coefficients_filtres = [(n, abs(c), np.angle(c)) for n, c in enumerate(coefficients)if abs(c) > 1e-13]
#print(coefficients_filtres)




#Dessin des cercles___________________________________________________________________________________

circles = [(0, -1.2412439092504478e-16), (-1, -2.0816681711721685e-17), (1, 1.0), (-2, -1.280691940746786e-16), (2, 0.0), (-3, -1.700029006457271e-16), (3, 0.4999999999999999)]
new_coeffficients[1:]
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
