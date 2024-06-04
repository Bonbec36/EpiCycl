import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def progress_callback(frame, total):
    if frame%5 == 0:
        print(f"Enregistrement du frame {frame+1} sur {total}", end="\r")



# Charger l'image
image = plt.imread('Pixel_image/002_tonystark_01.jpg')
print(sorted(list(image.shape)))

max_pixel = sorted(list(image.shape))[1]

# Initialiser la figure et l'axe
fig, ax = plt.subplots()
ax.set_title('Pixelisation Progressive')

# Définir les limites des axes
ax.set_xlim(0, image.shape[1])
ax.set_ylim(0, image.shape[0])

num_frames = 300

# Calculer les tailles de pixel à l'avance
step_factor = max_pixel/50_000
pixel_sizes = [int(0.5*max_pixel*np.exp(-step_factor*(frame+1))+1) for frame in range(num_frames)]

# Initialisation de l'image
im = ax.imshow(image, extent=(0, image.shape[1], 0, image.shape[0]), interpolation="spline36")

def update(frame):
    # Obtenir la taille du pixel pour cette frame
    pixel_size = pixel_sizes[frame]

    # Pixeliser l'image avec la taille de pixel calculée
    pixelated_image = np.repeat(np.repeat(image[::pixel_size, ::pixel_size], pixel_size, axis=1), pixel_size, axis=0)

    # Mettre à jour l'image existante avec les nouveaux pixels
    im.set_data(pixelated_image)

    return im,

# Créer l'animation
animation = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=200)

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'  # Chemin vers l'exécutable FFmpeg
plt.rcParams['savefig.dpi'] = 150

# Enregistrer l'animation au format mp4
Writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
animation.save(f'002_tonystark_01.mp4', writer=Writer, progress_callback=progress_callback)


# Afficher l'animation
plt.show()
