import imageio
import os

images = [os.path.join("dolly", f) for f in os.listdir("dolly") if f.endswith(".jpg")]

media = []
for filename in images:
    media.append(imageio.imread(filename))
imageio.mimsave('dollyzoom.gif', media)

