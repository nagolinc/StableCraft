import os
import glob

pngs=glob.glob("./static/samples/*.png")
for png in pngs:
    os.remove(png)

saves=glob.glob("./static/saveData/*")
for save in saves:
    if save!=".gitkeep":
        os.remove(save)