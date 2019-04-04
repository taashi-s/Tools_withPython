import os
import glob
import math
import cv2
import numpy as np
import PIL.Image as Image
from tqdm import tqdm


DATA_TYPE = 'valid'

BASE_DIR = os.path.join('.', 'origin')
ORIGIN_INPUTS_DIR = os.path.join(BASE_DIR, DATA_TYPE, 'inputs')
ORIGIN_TEACHERS_DIR = os.path.join(BASE_DIR, DATA_TYPE, 'teachers')

INPUTS_DIR = os.path.join('.', DATA_TYPE + '_inputs')
TEACHERS_DIR = os.path.join('.', DATA_TYPE + '_teachers')


def main():
    files = glob.glob(os.path.join(ORIGIN_INPUTS_DIR, '*.png'))
    files.sort()
    print('### target annotation file : ', len(files))
    print('')
    counts = []
    pbar = tqdm(total=len(files), desc="Augmentaion", unit=" Files")
    for file in files:
        file_name, _ = os.path.splitext(os.path.basename(file))
        teacher_path = os.path.join(ORIGIN_TEACHERS_DIR, file_name + '.png')
        input_img = cv2.imread(file)
        teacher_img = cv2.imread(teacher_path)
        if input_img is None:
            print('input_img is None : ', file)
            continue
        if teacher_img is None:
            print('teacher_img is None : ', teacher_path)
            continue
        count = image_data_augmentaiton(file_name, input_img, teacher_img
                                        #, flip_type=1
                                        #, scales=[0.8, 1.2]
                                        , rotate_stride=2
                                        , rotate_range=10
                                       )
        counts.append(count)
        pbar.update(1)
    pbar.close()

    if max(counts) == min(counts):
        print('')
        print('### augmentated all data : 1 --> ', max(counts))
    else:
        for count, file in zip(files, counts):
            file_name, _ = os.path.splitext(os.path.basename(file))
            print('### augmentate %s : 1 --> ' % file_name, count)




def image_data_augmentaiton(file_name, input_img, teacher_img
                            , flip_type=None, scales=None, rotate_stride=None, rotate_range=None):
    bases = []
    bases.append((file_name + '_N', input_img, teacher_img))
    if flip_type is not None:
        bases.append((file_name + '_F', cv2.flip(input_img, flip_type), cv2.flip(teacher_img, flip_type)))

    bases_2 = []
    if scales is None:
        scales = []
    for name, inp_img, tea_img in bases:
        bases_2.append((name + '_100', inp_img, tea_img))
        for scale in scales:
            bases_2.append((name + '_%03d' % (scale * 100), scaling_image(inp_img, scale), scaling_image(tea_img, scale)))

    bases_3 = []
    if rotate_stride is None:
        rotate_stride = 360
    if rotate_range is None:
        rotate_range = 180
    for name, inp_img, tea_img in bases_2:
        bases_3.append((name + '_000', inp_img, tea_img))
        for k in range((360 // rotate_stride) - 1):
            deg = rotate_stride * (k + 1)
            if rotate_range < deg and deg < 360 - rotate_range:
                continue
            bases_3.append((name + '_%03d' % deg, rotate_image(inp_img, deg), rotate_image(tea_img, deg)))

    data = bases_3
    for name, inp_img, tea_img in data:
        #print('data augmentation : ', name)
        input_save_path = os.path.join(INPUTS_DIR, name + '.png')
        teacher_save_path = os.path.join(TEACHERS_DIR, name + '.png')
        cv2.imwrite(input_save_path, inp_img)
        cv2.imwrite(teacher_save_path, tea_img)
#    print('### augmentation ', file_name, ' (1 -> ', len(data), ')')
    return len(data)


def scaling_image(img, scale):
    h, w, _ = np.shape(img)
    res_h, res_w = (math.floor(h * scale), math.floor(w * scale))
    img_resize = cv2.resize(img, (res_h, res_w))
    if scale < 1:
        img_scale = np.zeros(np.shape(img))
        pos_top = (h - res_h) // 2
        pos_left = (w - res_w) // 2
        img_scale[pos_top:pos_top+res_h,pos_left:pos_left+res_w,:] = img_resize
#        h_tile_num = __calculate_tile_num(h, res_h)
#        w_tile_num = __calculate_tile_num(w, res_w)
#        img_tile = np.tile(img_resize, (h_tile_num, w_tile_num, 1))
#        img_scale = __crop_image_center(img_tile, h, w)
    elif scale > 1:
        img_scale = __crop_image_center(img_resize, h, w)
    else:
        img_scale = img_resize
    return img_scale


def __calculate_tile_num(org_size, res_size):
    base = org_size / res_size
    if base - int(base) == 0:
        base = int(base)
        if base % 2 == 0:
            return base + 1
        else:
            return base
    else:
        return math.floor(base) + 2


def rotate_image(img, degree):
    img_tile = img #np.tile(img, (3, 3, 1))
    pil_img_tile = Image.fromarray(np.uint8(img_tile))
    pil_img_tile_rotate = pil_img_tile.rotate(degree)
    img_tile_rotate = np.array(pil_img_tile_rotate)

    img_rotate = img_tile_rotate # __crop_image_center(img_tile_rotate, *(np.shape(img)[:2]))
    return img_rotate


def __crop_image_center(img, crop_h, crop_w):
    h, w = np.shape(img)[:2]
    pos_top = (h - crop_h) // 2
    pos_left = (w - crop_w) // 2
    return img[pos_top:pos_top+crop_h,pos_left:pos_left+crop_w,:]


if __name__ == '__main__':
    main()
