import os
import sys
import logging
import argparse
import glob

import numpy as np
import openslide
import cv2
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and '
                                             'save it in npy format')
parser.add_argument('--wsi_path', default='/home/sdc/pangguang/svs', metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('--json_path', default='/home/sdd/zxy/TCGA_data/json/all_tumor_json_10_20', metavar='JSON_PATH', type=str,
                    help='Path to the JSON file')
parser.add_argument('--npy_path', default='/home/sdd/zxy/TCGA_data/npy_tumor/all_tumor_1020', metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')


def run(args):
    # paths = glob.glob(os.path.join(args.json_path, '*.json'))
    ff = os.walk(args.wsi_path)
    paths = []
    for root, dirs, files in ff:
        for file in files:
            if os.path.splitext(file)[1] == '.svs':
                paths.append(os.path.join(root, file))
    count = 0
    for path in paths:
        #import pdb;pdb.set_trace()
        wsi_name = os.path.basename(path)[:-4]
        json_path = os.path.join(args.json_path, wsi_name + '.json')
        npy_path = os.path.join(args.npy_path,wsi_name+'.npy')
        #if os.path.exists(npy_path):
        #    continue
        if not os.path.exists(json_path):
            continue
    # get the level * dimensions e.g. tumor0.tif level 6 shape (1589, 7514)
        slide = openslide.OpenSlide(path)
        print(json_path, count, slide.level_count)
        count +=1
        level_count = min(2,slide.level_count-1)
        w, h = slide.level_dimensions[level_count]
        mask_tumor = np.zeros((h, w)) # the init mask, and all the value is 0

        # get the factor of level * e.g. level 6 is 2^6
        factor = slide.level_downsamples[level_count]
        slide.close()
        with open(json_path) as f:
            dicts = json.load(f)
        tumor_polygons = dicts['positive']

        for tumor_polygon in tumor_polygons:
            # plot a polygon
            name = tumor_polygon["name"]
            vertices = np.array(tumor_polygon["vertices"]) / factor
            vertices = vertices.astype(np.int32)

            cv2.fillPoly(mask_tumor, [vertices], (255))

        mask_tumor = mask_tumor[:] > 127
        mask_tumor = np.transpose(mask_tumor)
        print(npy_path)
        np.save(npy_path, mask_tumor)

def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
