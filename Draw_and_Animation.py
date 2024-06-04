import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random


def read_complex_tuples_from_csv(filename):
    complex_tuples = []
    with open(filename, 'r') as csvfile:
        for line in csvfile:
            parts = line.split(',')  # Diviser chaque ligne en parties
            index = int(parts[0])  # Extraire l'indice
            real = float(parts[1])  # Extraire la partie réelle
            imag = float(parts[2])  # Extraire la partie imaginaire
            complex_tuples.append((index, complex(real, imag)))  # Ajouter le tuple à la liste
    return complex_tuples

def reorder_fourier_coeffs(coeffs):
    # Trier les coefficients en fonction de l'ordre de leurs indices absolus
    reordered_coeffs = sorted(coeffs, key=lambda x: abs(x[0]))
    return reordered_coeffs

def progress_callback(frame, total):
    if frame%5 == 0:
        print(f"Enregistrement du frame {frame+1} sur {total}", end="\r")



filename = 'Coeffs_024_shrek.csv'

indice_color = random.randint(0, 6)

liste_nb_circles = [5, 10, 25, 50, 75] #[5, 15, 40, 80, 120]

for nb_circles in liste_nb_circles:

    print(f"Nombre de cercle : {nb_circles}\n")

    
    print(indice_color)

    coeffs = read_complex_tuples_from_csv(filename)

    organized_coefficients = reorder_fourier_coeffs(coeffs)

    circles_parameter = [(o_c[0], abs(o_c[1]), np.angle(o_c[1])) for o_c in organized_coefficients]

    circles_parameter_non_zero = [c for c in circles_parameter[1:] if c[1] > 1e-10]


    print(len(circles_parameter))

    print(len(circles_parameter_non_zero))




    circles =  circles_parameter_non_zero[:nb_circles]
    nb_points = 500

    theta = np.linspace(0, 2*np.pi, nb_points)


    all_x = [circles[i][1]*np.cos(theta*circles[i][0]+circles[i][2]) for i in range(len(circles))]
    all_y = [circles[i][1]*np.sin(theta*circles[i][0]+circles[i][2]) for i in range(len(circles))]



    cord_circles_x = [sum(all_x[:i]) for i in range(1, len(all_x)+1)]
    cord_circles_y = [sum(all_y[:i]) for i in range(1, len(all_y)+1)]

    bordures = [min(np.array(cord_circles_x).flatten()), max(np.array(cord_circles_x).flatten()),
    min(np.array(cord_circles_y).flatten()), max(np.array(cord_circles_y).flatten())]

    x = cord_circles_x[-1]
    y = cord_circles_y[-1]

    """
    x_reconstruite = 2.5174897919009878*np.exp(2j*np.pi*t_values) 
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.plot(t_values, np.real(r_curve), "b-")
    #ax1.plot(theta[:250]/(2*np.pi), x[:250], "r-")
    ax1.plot(t_values, np.real(x_reconstruite), "g-")
    """

    draw_circle = [plt.Circle((0, 0), circles[i][1], color='gray', alpha=0.3) for i in range(len(circles))]



    def init():
        for circle2draw in draw_circle:
            ax.add_artist(circle2draw)
        ax.set_aspect('equal')
        return draw_circle,


    def animate(t):
        line.set_data(x[:t], y[:t])
        line_fluo.set_data(x[:t], y[:t])
            
        for i in range(1, len(draw_circle)):
            draw_circle[i].center = (cord_circles_x[i-1][t], cord_circles_y[i-1][t])
        scat.set_offsets(np.column_stack(([c.center[0] for c in draw_circle], [c.center[1] for c in draw_circle]))) 
        line_center.set_data([c.center[0] for c in draw_circle], [c.center[1] for c in draw_circle])
        last_scat.set_offsets(np.column_stack((cord_circles_x[-1][t], cord_circles_y[-1][t])))
        return line,draw_circle, scat, last_scat

    fig, ax = plt.subplots(figsize=(6, 6))

    bordures = [1]

    ax.set_xlim(-1.05*max(bordures), 1.05*max(bordures))
    ax.set_ylim(-1.05*max(bordures), 1.05*max(bordures))


    color_liste = ["#0FFF00", "#C100FF", "#002BFF", "#F3FF00", "#00F7FF", "#FFC900", "#FB00FF"]


    color_choice = color_liste[indice_color]

    line, = ax.plot([], [], color=color_choice, lw=2)
    line_fluo, = ax.plot([], [], color=color_choice, lw=6, alpha=0.3)
    line_center, = ax.plot([], [], color='#C8C8C8', linestyle='--', lw=2)

    scat = ax.scatter([c.center[0] for c in draw_circle], [c.center[1] for c in draw_circle], color="#FF0000", s=10)
    last_scat = ax.scatter(0, 0, color='purple', marker='D', s=10)

    plt.title(f"{nb_circles} circles", fontsize=20, color="white")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.set_facecolor('black')
    ax.set_facecolor('black')


    ani = FuncAnimation(fig, animate, frames=range(nb_points), init_func=init,blit=True, interval=10)

    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'  # Chemin vers l'exécutable FFmpeg
    plt.rcParams['savefig.dpi'] = 150

    # Enregistrer l'animation au format mp4
    Writer = FFMpegWriter(fps=40, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(f'{filename[7:-4]}_{nb_circles:02d}.mp4', writer=Writer, progress_callback=progress_callback)

    #plt.show()
import subprocess

def concatenate_videos(video_files, output_file):
    # Créer un fichier temporaire contenant la liste des vidéos à concaténer
    with open('list.txt', 'w') as f:
        for file in video_files:
            f.write(f"file '{file}'\n")
    
    # Concaténer les vidéos en utilisant ffmpeg
    subprocess.run([
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'list.txt', '-c', 'copy', output_file
    ])

    # Supprimer le fichier temporaire
    subprocess.run(['rm', 'list.txt'])

# Exemple d'utilisation
video_files = [f"{filename[7:-4]}_{liste_nb_circles[0]:02d}.mp4", f"{filename[7:-4]}_{liste_nb_circles[1]:02d}.mp4",
                f"{filename[7:-4]}_{liste_nb_circles[2]:02d}.mp4", f"{filename[7:-4]}_{liste_nb_circles[3]:02d}.mp4", 
                f"{filename[7:-4]}_{liste_nb_circles[4]:02d}.mp4"]
output_file = f"{filename[7:10]}_Full_{filename[11:-4]}.mp4"
concatenate_videos(video_files, output_file)
