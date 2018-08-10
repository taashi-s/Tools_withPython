import os
import glob
import json
import cv2
import numpy as np
import pprint
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

IMAGE_DIR = os.path.join('.', 'Inputs')
ANNOTAION_DIR = os.path.join('.', 'Outputs', 'annotations')
MASK_OUTPUT_DIR = os.path.join('.', 'Outputs', 'mask')
SEGMENTATION_DIR = os.path.join('.', 'Outputs', 'segmentation')
OVERLAY_DIR = os.path.join('.', 'Outputs', 'overlay')


def main():
    files = glob.glob(os.path.join(ANNOTAION_DIR, '*.json'))
    files.sort()
    print('### target annotation file : ', len(files))
    print('')
    for file in files:
        create_annotation_img(file)


def create_annotation_img(anno_json):
    jf = json.load(open(anno_json))
    image_name_base, _ = os.path.splitext(os.path.basename(anno_json))

    original_image_path = os.path.join(IMAGE_DIR, image_name_base)
    org_img = cv2.imread(original_image_path + '.png')
    if org_img is None:
        org_img = cv2.imread(original_image_path + '.jpeg')
    org_img = org_img.astype(np.uint8)
    image_shape = np.shape(org_img)

    seg_img = None
    for k, shape in enumerate(jf['shapes']):
        contours = shape['points']

        mask = contours
        if len(mask) < 2:
            print('mask region is None : ', image_name_base, '[%2d]' % k)
            continue
        img = create_mask_image(image_shape, mask)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if shape['label'] == 'Crack':
            img[:,:,1:] = 0
        elif shape['label'] == 'Explosion':
            img[:,:,0] = 0
            img[:,:,1] = 0
        else:
            # skip
            continue

        file_name = '%s_%d.png' % (image_name_base, k)
        file_path = os.path.join(MASK_OUTPUT_DIR, file_name)
        cv2.imwrite(file_path, img)
        
        if seg_img is None:
            seg_img = img
        else:
            seg_img += img
            seg_img[seg_img > 255] = 255
    
    if seg_img is None:
        print('seg_img is None. : ', image_name_base)
        return

    seg_img = seg_img.astype(np.uint8)
    segmentation_path = os.path.join(SEGMENTATION_DIR, image_name_base + '.png')
    seg_img_resize = cv2.resize(seg_img, (256, 256))
    cv2.imwrite(segmentation_path, seg_img_resize)


    overlay_ing = cv2.addWeighted(org_img, 1, seg_img, 0.6, 0)
    overlay_path = os.path.join(OVERLAY_DIR, image_name_base + '.png')
    overlay_ing_resize = cv2.resize(overlay_ing, (256, 256))
    cv2.imwrite(overlay_path, overlay_ing_resize)
    print(segmentation_path)


def create_mask_image(image_shape, mask):
    img_src = np.zeros(image_shape[:2])
    img = Image.fromarray(img_src)
    xy = list(map(tuple, mask))
    ImageDraw.Draw(img).polygon(xy=xy, outline=1, fill=1)
    img = np.array(img) * 255
    return img


if __name__ == '__main__':
    main()
