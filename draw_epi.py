import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d


circles = [(1, 1, 0), (-1, 0.4, 1), (2, 0.1, 3)]
nb_points = 100

theta = np.linspace(0, 2*np.pi, nb_points)


all_x = [circles[i][1]*np.cos(theta*circles[i][0]+circles[i][2]) for i in range(len(circles))]
all_y = [circles[i][1]*np.sin(theta*circles[i][0]+circles[i][2]) for i in range(len(circles))]



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
