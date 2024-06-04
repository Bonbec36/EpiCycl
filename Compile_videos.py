import subprocess

filename = 'Coeffs_018_ferrari.csv'

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

video_files = [f"{filename[7:-4]}_05.mp4", f"{filename[7:-4]}_15.mp4",
                f"{filename[7:-4]}_40.mp4", f"{filename[7:-4]}_80.mp4", f"{filename[7:-4]}_120.mp4"]
output_file = f"{filename[7:10]}_Full_{filename[11:-4]}.mp4"
concatenate_videos(video_files, output_file)
