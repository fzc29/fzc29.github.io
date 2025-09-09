import imageio
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
dolly_dir = os.path.join(base_dir, "dolly")  

images = [os.path.join(dolly_dir, f) for f in os.listdir(dolly_dir) if f.endswith(".jpg")]

media = []
for filename in images:
    media.append(imageio.imread(filename))
imageio.mimsave('dollyzoom.gif', media, loop=0)


# cite: stackoverflow: https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python