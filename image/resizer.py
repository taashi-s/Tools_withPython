
from PIL import Image
import numpy as np
import os
import glob

RSIZE_SIZE = (128, 128)

for file in glob.glob(os.path.join("Inputs", "*.png")):
    print(file)
    srcImg = Image.open(file)
    distImg = srcImg.resize(RSIZE_SIZE)
    (name, _) = os.path.splitext((os.path.basename(file)))
    distImg.save(os.path.join("Outputs", name + ".png"), "png")

print("finish !")
