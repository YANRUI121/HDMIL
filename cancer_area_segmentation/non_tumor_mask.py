import sys
import os
import argparse
import logging
import glob

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

parser = argparse.ArgumentParser(description="Get the normal region"
                                             " from tumor WSI ")
parser.add_argument("--tumor_path", default='/home/sdd/zxy/TCGA_data/npy_tumor_jingbiaozhu_0318/train/wild', metavar='TUMOR_PATH', type=str,
                    help="Path to the tumor mask npy")
parser.add_argument("--tissue_path", default='/home/sdd/zxy/TCGA_data/npy_tissue/wild', metavar='TISSUE_PATH', type=str,
                    help="Path to the tissue mask npy")
parser.add_argument("--normal_path", default='/home/sdd/zxy/TCGA_data/npy_normal_jingbiaozhu_0318/train/wild', metavar='NORMAL_PATCH', type=str,
                    help="Path to the output normal region from tumor WSI npy")


def run(args):
    paths = glob.glob(os.path.join(args.tumor_path, '*.npy'))
    for path in paths:
        print(path)
        tumor_mask = np.load(path)
        tissue_name = os.path.basename(path)
        tissue_path = os.path.join(args.tissue_path,tissue_name)
        tissue_mask = np.load(tissue_path)

        normal_mask = tissue_mask & (~ tumor_mask)
        normal_path = os.path.join(args.normal_path,tissue_name)
        np.save(normal_path, normal_mask)

def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
