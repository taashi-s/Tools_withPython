
from PIL import Image
from PIL import ImageFilter
import os
import glob


for file in glob.glob(os.path.join("Inputs", "*.png")):
    print(file)
    srcImg = Image.open(file)
    distImg = srcImg
    distImg = distImg.filter(ImageFilter.FIND_EDGES)
    distImg = distImg.filter(ImageFilter.MaxFilter())
    distImg = distImg.filter(ImageFilter.MaxFilter())
    distImg = distImg.filter(ImageFilter.MinFilter())
    distImg = distImg.convert("L")
    distImg = distImg.point(lambda x: 0 if x < 200 else 255)
    # distImg = distImg.convert("1")
    (name, _) = os.path.splitext((os.path.basename(file)))
    distImg.save("Outputs/" + name + ".png", "png")

print("finish !")
