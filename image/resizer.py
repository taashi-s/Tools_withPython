#from PIL import Image
import cv2
import numpy as np
import os
import glob
#import time

RSIZE_SIZE = (256, 256)

ORIGINAL_DIR = os.path.join('.', 'Inputs')
OUTPUT_DIR = os.path.join('.', 'Outputs')

files = []
files += glob.glob(os.path.join(ORIGINAL_DIR, "*.jpeg"))
files += glob.glob(os.path.join(ORIGINAL_DIR, "*.jpg"))
files += glob.glob(os.path.join(ORIGINAL_DIR, "*.png"))
files.sort()

print('resizing ...', end='', flush=True)
#s_time = time.time()
for file in files:
    #print(file)
    (name, _) = os.path.splitext((os.path.basename(file)))
    resize_file_path = os.path.join(OUTPUT_DIR, name + ".png")

    #srcImg = Image.open(file)
    #distImg = srcImg.resize(RSIZE_SIZE)
    #distImg.save(resize_file_path, "png")

    img = cv2.imread(file)
    img_resize = cv2.resize(img, RSIZE_SIZE)
    cv2.imwrite(resize_file_path, img_resize)
#e_time = time.time()
print("finish !")
#print('total time : ', e_time - s_time)
#print('time by one image : ', (e_time - s_time) / len(files))
