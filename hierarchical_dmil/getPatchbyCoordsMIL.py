import pandas as pd
import torchvision.transforms as transforms
import torch
import numpy as np
import argparse
import glob
import os
import random
import openslide
import cv2

parser = argparse.ArgumentParser(description='K-menas for json file')
parser.add_argument('--txt_path', default='/home/sdd/zxy/TCGA_data/selectTxt/tcga_2_all_cancer_0402/train/wild', metavar='TXT_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--wsi_path', default='/home/sdc/pg/svs/train/wild', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--img_path', default='/home/sdd/zxy/TCGA_data/mil_patch/train/1_wild', metavar='IMG_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--img_size', default=512, metavar='IMG_SIZE', type=int, help='Patch size')
parser.add_argument('--level', default=0, metavar='LEVEL', type=int, help='the level of wsi')
parser.add_argument('--rate', default=512, metavar='RATE', type=int, help='rate of the reduction of coords')


def cut_img(args, slide, coords, patch_dir):
    print('************Cuting Image***************')
    count_num = 0
    for coord in coords:
        img = slide.read_region(coord, args.level, (args.img_size, args.img_size)).convert('RGB')
        img.save(os.path.join(patch_dir, str(coord[0]) + '_' + str(coord[1]) + '_' + str(count_num) + '.png'))
        count_num += 1
    print( '************Cuting End***************' )


def str2int(str_coord):
    coord_x = int(str_coord[str_coord.find('(') + 1: str_coord.find(',')]) * 512
    coord_y = int(str_coord[str_coord.find(',') + 2: str_coord.find(')')]) * 512
    return (coord_x, coord_y)

def getCoordsbyTxt(txt_path):
    with open(txt_path, 'r') as f:
        coord_list = f.read().splitlines()  #remove line breaks
    coords = list(map(str2int, coord_list))
    return coords


def run(args):
    paths = glob.glob(os.path.join(args.txt_path, '*.txt'))
    num_set = set(list(range(100)))    #The mutation data is amplified by 6 times
    #num_set = set(list(range(6)))    #The mutation data is amplified by 36 times,and nonmutation data is amplified by 6 times
    for path in paths[::-1]:
        npy_name = os.path.basename(path)
        num_count = npy_name[npy_name.rfind('_')+1:npy_name.rfind('.')]
        #if num_count != '0':
        if int(num_count) not in num_set:
            continue
        print(path)
        wsi_name = npy_name[0:npy_name.find('kmeans')]
        wsi_path = os.path.join(args.wsi_path, wsi_name + '.svs')
        if not os.path.exists(wsi_path):
            continue
        patch_dir = os.path.join(args.img_path, wsi_name + '_' + num_count)
        if os.path.exists(patch_dir):
            continue
        else:
            os.mkdir(patch_dir)
            slide = openslide.OpenSlide(wsi_path)
            coords = getCoordsbyTxt(path)
            cut_img(args, slide, coords, patch_dir)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()


