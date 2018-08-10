
from PIL import Image
from PIL import ImageFilter
import numpy as np
import os
import glob


for file in glob.glob(os.path.join("Inputs", "*.png")):
    print(file)
    srcImg = Image.open(file)
    distImg = srcImg
    distImg = distImg.convert("L")
    distImg = distImg.point(lambda x: 0 if x < 100 else 255)
    # distImg = distImg.convert("1")
    (name, _) = os.path.splitext((os.path.basename(file)))
    distImg.save("Outputs/" + name + ".png", "png")

print("finish !")
