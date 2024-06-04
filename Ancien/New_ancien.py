import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import matplotlib.animation as anim

def sort_points(points):
  # pick a point
  reference_point = points[0]
  sorted = [reference_point]
  remaining_points = range(1,len(points))
  for i in range(1,len(points)):

    # find the closest point to reference_point,
    mindiff = np.sum(np.square(np.array(points[remaining_points[0]])-reference_point))
    idx = 0
    # loop over all the other remaining points
    for j in range(1,len(remaining_points)):
      diff = np.sum(np.square(np.array(points[remaining_points[j]])-reference_point))
      if diff < mindiff:
        mindiff = diff
        idx = j
    # found the closest: update the selected point, and add it to the list of sorted points
    reference_point = points[remaining_points[idx]]
    sorted.append(reference_point )
    remaining_points = np.delete(remaining_points, idx)
  return sorted

homer = [(0, 1), (1, 0)]
sorted_homer = sort_points(homer)
resol = 50
n_fourrier = 51
#6, 9, 13, 100

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

def complex_parametrization(points):
    # Convert the list of points to a numpy array
    points = np.array(points)

    # Compute the centroid of the points
    center = np.mean(points, axis=0)

    # Compute the complex coordinates of the points relative to the centroid
    z = (points[:, 0] - center[0]) + 1j * (points[:, 1] - center[1])


    return z

points = [[-1, -1], [-1, -3], [1, -3], [1, -1], [3, -1], [3, 1], [1, 1], [1, 3], [-1, 3], [-1, 1], [-3, 1], [-3, -1], [-1, -1]]

smooth_points = smooth_curve(points)


P = 1
BT = 0
ET = 2
FS = 1000

f = lambda t: ((t % P) - (P / 2.)) ** 3
t_range = np.linspace(BT, ET, FS)
y_true = f(t_range)

"""def compute_real_fourier_coeffs(func, N):
    result = []
    for n in range(N+1):
        an = (2./P) * spi.quad(lambda t: func(t) * np.cos(2 * np.pi * n * t / P), 0, P)[0]
        bn = (2./P) * spi.quad(lambda t: func(t) * np.sin(2 * np.pi * n * t / P), 0, P)[0]
        result.append((an, bn))
    return np.array(result)"""


#ensemble de points (x,y)
x = np.linspace(0,1, len(smooth_points))
y = np.array(smooth_points[:, 0])
z = np.array(smooth_points[:, 1])


#function that computes the real fourier couples of coefficients (a0, 0), (a1, b1)...(aN, bN)
def compute_real_fourier_coeffs(x, y, N):
    result = []
    for n in range(N+1):
        an = (2./P) * spi.simps(y * np.cos(2 * np.pi * n * x / P), x)
        bn = (2./P) * spi.simps(y * np.sin(2 * np.pi * n * x / P), x)
        result.append((an, bn))
    return np.array(result)

def fit_func_by_fourier_series_with_real_coeffs(t, AB):
    A = AB[:,0]
    B = AB[:,1]
    result = A[0]/2. + sum(A[n] * np.cos(2. * np.pi * n * t / P) + B[n] * np.sin(2. * np.pi * n * t / P) for n in range(1, len(AB)))
    return result

def compute_complex_fourier_coeffs_from_real(AB):
    N = len(AB) - 1
    coeffs = np.zeros(N + 1, dtype=np.complex128)
    coeffs[0] = AB[0, 0] / 2
    coeffs[1:] = (AB[1:, 0] + 1j * AB[1:, 1]) / 2
    return coeffs

def sort_complex_coeffs(coeffs):
    N = (len(coeffs) - 1) // 2
    sorted_coeffs = np.zeros_like(coeffs, dtype=np.complex128)
    sorted_coeffs[0] = coeffs[0]  # Coefficient constant
    sorted_coeffs[1] = coeffs[1]  # Coefficient pour n=1

    for n in range(1, N + 1):
        sorted_coeffs[2*n] = coeffs[n]  # Coefficient pour n
        sorted_coeffs[2*n+1] = coeffs[-n]  # Coefficient pour -n
    
    return sorted_coeffs

maxN = 1
COLs = 1
ROWs = 1
plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(ROWs, COLs)

AB = compute_real_fourier_coeffs(x, y, n_fourrier)
AB_1 = compute_real_fourier_coeffs(x, z, n_fourrier)
y_approx = fit_func_by_fourier_series_with_real_coeffs(t_range, AB)
z_approx = fit_func_by_fourier_series_with_real_coeffs(t_range, AB_1)


#axs.plot(x, y, label="y")
#axs.plot(x, z, label="z")

error_margin = np.ceil(len(y_approx)/10)



#axs.plot(t_range, y_approx, label="y approx")
#axs.plot(t_range, z_approx, label="z approx")


axs.plot(y_approx, z_approx, alpha=0)
#axs.plot([p[0] for p in sorted_homer], [p[1] for p in sorted_homer])



line, = axs.plot([], [], color='blue')
point, = axs.plot([], [], ls="none", marker="o")

AB_2 = compute_real_fourier_coeffs(x, y, 50)
AB_3 = compute_real_fourier_coeffs(x, z, 50)
y_approx_1 = fit_func_by_fourier_series_with_real_coeffs(t_range, AB_2)
z_approx_1 = fit_func_by_fourier_series_with_real_coeffs(t_range, AB_3)



fourrier_fontion_by_indices = [[fit_func_by_fourier_series_with_real_coeffs(t_range, compute_real_fourier_coeffs(x, y, n_fourrier+5*i)), fit_func_by_fourier_series_with_real_coeffs(t_range, compute_real_fourier_coeffs(x, z, n_fourrier+5*i))] for i in range(5)]


# Création de la function qui sera appelée à "chaque nouvelle image"
def animate(k):
    """
    if k//len(y_approx) == 0:
        line.set_data(y_approx[:k], z_approx[:k])
    #point.set_data(y_approx[i], z_approx[i])
    elif k//len(y_approx) == 1:
        line.set_data(y_approx_1[:k-len(y_approx_1)], z_approx_1[:k-len(z_approx_1)])
    """
    line.set_data(fourrier_fontion_by_indices[k//len(y_approx)][0][:k-(len(y_approx)*(k//len(y_approx)))], fourrier_fontion_by_indices[k//len(y_approx)][1][:k-(len(y_approx)*(k//len(y_approx)))])
    return line,


# Génération de l'animation, frames précise les arguments numérique reçus par func (ici animate),
# interval est la durée d'une image en ms, blit gère la mise à jour
ani = anim.FuncAnimation(fig=fig, func=animate, frames=len(y_approx)*len(fourrier_fontion_by_indices), interval=1, blit=True, repeat=False)
plt.show()


coef_sin = AB[:, 0]
coef_cos = AB[:, 1]

print(AB[:5])



def convert_to_epicycloids(coef_sin, coef_cos, omega):
    N = len(coef_sin)
    radii = np.zeros(N)
    phases = np.zeros(N)
    angular_velocities = np.zeros(N)

    for n in range(N):
        radius = np.sqrt(coef_sin[n]**2 + coef_cos[n]**2)
        phase = np.arctan2(coef_cos[n], coef_sin[n])
        angular_velocity = n * omega
        radii[n] = radius
        phases[n] = phase
        angular_velocities[n] = angular_velocity

    return radii, angular_velocities, phases

def convert_to_epicycloids(coefficients):
    N = len(coefficients)
    radii = np.zeros(N)
    phases = np.zeros(N)
    angular_velocities = np.zeros(N)

    for n in range(N):
        # Rayon
        radius = np.sqrt(coefficients[n, 0]**2 + coefficients[n, 1]**2)
        radii[n] = radius

        # Phase
        phase = np.arctan2(coefficients[n, 1], coefficients[n, 0])
        phases[n] = phase

        # Vitesse angulaire
        angular_velocity = n + 1  # N'oubliez pas d'ajuster si nécessaire
        angular_velocities[n] = angular_velocity

    return radii, phases, angular_velocities

# Convertir les coefficients en paramètres de cercles épicycloïdes
radii, phases, angular_velocities = convert_to_epicycloids(AB)

# Afficher les paramètres des cercles épicycloïdes
for i in range(len(radii)):
    print(f"Cercle {i+1}: R={radii[i]}, phi={phases[i]}, omega={angular_velocities[i]}")

