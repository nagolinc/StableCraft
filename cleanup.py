#

import os
import glob

pngs=glob.glob("./static/samples/*.png")
for png in pngs:
    os.remove(png)
    
    
#remove mp3's
for mp3 in glob.glob("./static/samples/*.mp3"):
    os.remove(mp3)

saves=glob.glob("./static/saveData/*")
for save in saves:
    if save!=".gitkeep":
        os.remove(save)

db=glob.glob("./mydatabase*")        
for f in db:
    os.remove(f)        