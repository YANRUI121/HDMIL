import os
import argparse
import time
import logging
import numpy as np
import openslide
import random
from multiprocessing import Pool, Value, Lock

parser = argparse.ArgumentParser(description='Generate the patch of tumor')
parser.add_argument('--wsi_path', default=r'/home/sdc/fuzhong-linchuang/svs/train/wild', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--mask_path', default=r'/home/sdc/fuzhong-linchuang/midFiles/tumor/valid/wild', metavar='MASK_PATH', type=str,
                    help='Path to the tumor mask of the input WSI file')
parser.add_argument('--patch_path', default='/home/sdd/zxy/TCGA_data/train_valid_256_level1_linchuang/valid/1_tumor', metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                    'default 768')
parser.add_argument('--level', default=1, type=int, help='patch size, '
                    'default 1')
parser.add_argument('--num_process', default=10, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()

def process(opts):
    i, wsi_path, x_center, y_center, rate, args = opts
    x = int(int(x_center) - args.patch_size * rate / 2)
    y = int(int(y_center) - args.patch_size * rate / 2)
    slide = openslide.OpenSlide(wsi_path)
    img = slide.read_region(
        (x, y), args.level,
        (args.patch_size, args.patch_size)).convert('RGB')

    img.save(os.path.join(args.patch_path, str(i) + '_w0817.png'))

    global lock
    global count

    with lock:
        count.value += 1
        if (count.value) % 100 == 0:
            logging.info('{}, {} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 count.value))


def run(args):
    logging.basicConfig(level=logging.INFO)
    ff = os.walk(args.wsi_path)
    paths = []
    opts_list = []
    for root, dirs, files in ff:
        for file in files:
            if os.path.splitext(file)[1] == '.svs':
                paths.append(os.path.join(root, file))
    num = 1000 
    num_svs = 0
    for path in paths:
        wsi_name = os.path.basename(path)[:-4]
        mask_path = os.path.join(args.mask_path, wsi_name + '.npy')
        if not os.path.exists(mask_path):
            continue
        mask = np.load(mask_path)
        slide = openslide.OpenSlide(path)
        # level = slide.level_count
        X_slide, Y_slide = slide.level_dimensions[0]
        x_level, y_level = slide.level_dimensions[args.level]
        rate = X_slide // x_level
        print(path, slide.level_count,rate, X_slide, Y_slide)
        slide.close()
        X_mask, Y_mask = mask.shape

        if X_slide // X_mask != Y_slide // Y_mask:
            raise Exception('Slide/Mask dimension does not match ,'
                            'X_slide / X_mask: {} / {}, '
                            'Y_slide / Y_mask: {} / {}'
                            .format(X_slide, X_mask, Y_slide, Y_mask))


        resolution = X_slide // X_mask
        # print(path, level, X_slide, Y_slide,X_mask, Y_mask, resolution)
        resolution_log = np.log2(resolution)
        if not resolution_log.is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2: '
                            '{}'.format(resolution))

        # all the idces for tissue region from the tissue mask
        X_idcs, Y_idcs = np.where(mask)
        coords = []
        for idc_num in range(len(X_idcs)):
            temp = [X_idcs[idc_num], Y_idcs[idc_num]]
            coords.append(temp)
        random.shuffle(coords)
        print(path, len(X_idcs), resolution, num_svs)
        num_svs += 1
        #import pdb;pdb.set_trace()
        for idx in range(min(1000,len(coords))):
                x_mask, y_mask = coords[idx][0], coords[idx][1]
                x_center = int((x_mask + 0.5) * resolution)
                y_center = int((y_mask + 0.5) * resolution)
                opts_list.append((num, path, x_center, y_center, rate, args))
                num = num + 1
    print(len(opts_list))
    pool = Pool(processes=args.num_process)
    pool.map(process, opts_list)



def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
