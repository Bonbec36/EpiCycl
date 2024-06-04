import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp_integ
import scipy.interpolate as sp_inter
from matplotlib.animation import FuncAnimation

def smooth_curve(points, resolution=10):
    # Convert the list of points to a numpy array
    points = np.array(points)

    # Compute the distance between consecutive points
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    d = np.sqrt(dx**2 + dy**2)

    # Compute the total length of the curve
    length = np.sum(d)

    # Compute the number of segments to use
    segments = int(length * resolution+1)

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

    smoothed_curve[-1] = points[0]
    return smoothed_curve

fleche = [(0, -2), (2, 2), (0, 1), (-2, 2), (0, -2)]
ruban = [(1, 5), (0, 1), (-1, 5), (-1, -2), (1, -2), (1, 5)]
sqooch = [(1.7, 3.3), (1.4, 2.3), (1.7, 1.7), (2.4, 1.4), (3.2, 1.7), (4.2, 1.9), (5.1, 2.2),
          (6, 2.4), (6.9, 2.7), (7.8, 2.9), (9.2, 3.2), (7.5, 2.5), (6.4, 2),
          (5.1, 1.5), (4, 1), (2.7, 0.5), (1.6, 0.2), (0.7, 0.5), (0.5, 1.2),
          (0.7, 1.9), (1.2, 2.6), (1.7, 3.3)]
triangle = [(0, 2), (-2, 0), (2, 0), (0, 2)]
cross_shape = [(3, 1), (1, 1), (1, 3), (-1, 3), (-1, 1), (-3, 1), (-3, -1), (-1, -1), (-1, -3), (1, -3), (1, -1), (3, -1), (3, 1)]
brut_squarre = [(1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)]
point_to_smooth = smooth_curve(cross_shape, resolution=10)
points = np.array(point_to_smooth)

def z(t):
    # Interpolation des parties réelle et imaginaire séparément
    x_interp = sp_inter.interp1d(np.linspace(0, 1, len(points)), points[:, 0], kind='linear')(t)
    y_interp = sp_inter.interp1d(np.linspace(0, 1, len(points)), points[:, 1], kind='linear')(t)
    return x_interp + 1j * y_interp

# Générer des valeurs de t
t_values = np.linspace(0, 1, len(point_to_smooth))

# Calculer les valeurs de la fonction complexe pour chaque valeur de t
z_values = z(t_values)

def compute_complex_fourier_coeffs(z, n):
    coeffs = []
    for k in range(-n, n+1):
        ck_real = sp_integ.quad(lambda t: np.real(z(t)) * np.cos(2 * np.pi * k * t) +
                                       np.imag(z(t)) * np.sin(2 * np.pi * k * t), 0, 1)[0]
        ck_imag = sp_integ.quad(lambda t: -np.real(z(t)) * np.sin(2 * np.pi * k * t) +
                                       np.imag(z(t)) * np.cos(2 * np.pi * k * t), 0, 1)[0]
        ck = ck_real - 1j * ck_imag
        coeffs.append((k, ck))
    return coeffs


def reconstruct_curve_from_fourier_coeffs(coeffs, x_values):
    result = np.zeros_like(x_values, dtype=np.complex128)
    for n, cn in coeffs:
        result += cn * np.exp(2j * np.pi * n * x_values)
    return result

def reorder_fourier_coeffs(coeffs):
    # Trier les coefficients en fonction de l'ordre de leurs indices absolus
    reordered_coeffs = sorted(coeffs, key=lambda x: abs(x[0]))
    return reordered_coeffs


# Paramètres
n_values = 5  # Nombre de coefficients à calculer
M = 2*n_values+1
T = 1          # Période

# Calcul des coefficients complexes
coeffs = compute_complex_fourier_coeffs(z, n_values)



r_curve = reconstruct_curve_from_fourier_coeffs(coeffs, t_values)
#r_curve_fft = reconstruct_curve_from_fourier_coeffs(coeffs_fft_list, t_values)

# Affichage des coefficients

"""for n, cn in coeffs:
    print(f"c_{n} = {cn}")
"""

# Tracer la partie réelle et la partie imaginaire de la fonction complexe
plt.plot(np.real(z_values), np.imag(z_values), 'b-', label='courbe fermée')
plt.plot(np.real(r_curve), np.imag(r_curve), 'r-', label='Courbe reconstruite')

#plt.plot(t_values, np.real(r_curve_fft), 'g-', label='Courbe reconstruite FFT')
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Tracé de la fonction complexe fermée, continue et lisse')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Assure l'échelle égale sur les axes x et y
#plt.show()

organized_coefficients = reorder_fourier_coeffs(coeffs)

circles_parameter = [(o_c[0], abs(o_c[1]), np.angle(o_c[1])) for o_c in organized_coefficients]

circles_parameter_non_zero = [c for c in circles_parameter[1:] if c[1] > 1e-10]


print(circles_parameter)

print(circles_parameter_non_zero)

new_cpnz = [i for i in circles_parameter_non_zero[:-1]]
new_cpnz.append((7, 0.1))

circles =  circles_parameter_non_zero
nb_points = 500

theta = np.linspace(0, 4*np.pi, nb_points)


all_x = [circles[i][1]*np.cos(theta*circles[i][0]+circles[i][2]) for i in range(len(circles))]
all_y = [circles[i][1]*np.sin(theta*circles[i][0]+circles[i][2]) for i in range(len(circles))]



cord_circles_x = [sum(all_x[:i]) for i in range(1, len(all_x)+1)]
cord_circles_y = [sum(all_y[:i]) for i in range(1, len(all_y)+1)]

bordures = [min(np.array(cord_circles_x).flatten()), max(np.array(cord_circles_x).flatten()),
min(np.array(cord_circles_y).flatten()), max(np.array(cord_circles_y).flatten())]

x = cord_circles_x[-1]
y = cord_circles_y[-1]

x_reconstruite = 2.5174897919009878*np.exp(2j*np.pi*t_values) 
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.plot(t_values, np.real(r_curve), "b-")
#ax1.plot(theta[:250]/(2*np.pi), x[:250], "r-")
ax1.plot(t_values, np.real(x_reconstruite), "g-")


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
