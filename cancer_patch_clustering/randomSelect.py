import pandas as pd
import numpy as np
import argparse
import glob
import os
import random
import openslide
import cv2

parser = argparse.ArgumentParser(description='K-menas for json file')
# parser.add_argument('--json_path', default = r'C:\ZXY\record\mutation_json', metavar='WSI_PATH', type=str,
#                     help='Path to the input WSI file')
parser.add_argument('--txt_path', default='/home/sdd/zxy/TCGA_data/all_clustering/FGFR3', metavar='TXT_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--txt_res_path', default='/home/sdd/zxy/TCGA_data/all_select_txt/FGFR3', metavar='TXT_RES_PATH', type=str,
                    help='Path to the input WSI file')
#parser.add_argument('--wsi_path', default='/home/sdc/fuzhong-linchuang/svs/wild', metavar='WSI_PATH', type=str,
#                    help='Path to the input WSI file')
#parser.add_argument('--img_path', default='/home/sdc/fuzhong-linchuang/midFiles/selectPng3/wild', metavar='IMG_PATH', type=str,
#                    help='Path to the input WSI file')
parser.add_argument('--cls_num', default='50', metavar='CLS_NUM', type=int,help='Class Number')

random = np.random.RandomState(0)

def get_wsi_path(args, txt_name):
    dir_pos = txt_name.find('kmean')
    wsi_path = os.path.join(args.wsi_path,txt_name[:dir_pos]+'.svs')
    #dir_pos = txt_name.find('00')
    #ff = os.walk(os.path.join(os.path.join(args.wsi_path, txt_name[:dir_pos - 1]),txt_name[:dir_pos + 6]))
    #ff = os.walk(os.path.join(os.path.join(args.wsi_path, txt_name[:dir_pos - 1]),txt_name[:dir_pos + 6]))
    #paths = []
    #for root, dirs, files in ff:
    #    for file in files:
    #        if os.path.splitext(file)[1] == '.svs':
    #            paths.append(os.path.join(root, file))
    return wsi_path

def cut_img(args, path, ext, coords):
    img_size = 768
#    import pdb;pdb.set_trace()
    txt_name = os.path.basename(path)
    wsi_path = get_wsi_path(args, txt_name)
   # assert len(wsi_path_list)==1,'Too much svs OR too less svs'
    #for wsi_path in wsi_path_list:
    slide = openslide.open_slide(wsi_path)
    img_name = os.path.basename(wsi_path[:-4])
    img_dir = os.path.join(args.img_path, img_name + '_' + str(ext))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    print('Cuting ' + wsi_path)
    assert len(coords)==args.cls_num,'Wrong numbers of patches'
    count_num = 0
    for coord in coords:
        count_num += 1
        coord = coord.split(',')
        x = int(coord[0][1:])
        y = int(coord[1][:-1])
        img = slide.read_region((x*img_size, y*img_size), 1, (img_size, img_size)).convert('RGB')
        img.save(os.path.join(img_dir, img_name + '_' + str(x) + '_' + str(y) + '_' + str(count_num) + '.png'))


def run(args):
    #train_type = ['valid', 'test']
    train_type = ['train', 'valid', 'test']
    mutation_type = ['mutation', 'wild']
    for train_name in train_type:
        for mutation_name in mutation_type:
            txt_res_path = os.path.join(os.path.join(args.txt_res_path, train_name), mutation_name)
            paths = glob.glob(os.path.join(os.path.join(os.path.join(args.txt_path,train_name), mutation_name), '*.txt'))
            for path in paths:
                print(path)
                basename = os.path.basename(path)
                #png_name = basename[:basename.find('kmean')]
                png_name = basename[:-4]
                #if os.path.exists(os.path.join(args.img_path,png_name)):
                #    continue
                file_object = open(path)
                try:
                    file_content = file_object.read()
                finally:
                    file_object.close()
                a = {}
                results = file_content.split('\n')
                results = list(filter(None, results))
                for i in range(len(results)):
                    coord_cls = results[i].split('\t')
                    if int(coord_cls[1]) in a:
                        a[int(coord_cls[1])].append(coord_cls[0])
                    else:
                        a[int(coord_cls[1])] = [coord_cls[0]]
                #if train_name == 'train' and mutation_name == 'mutation':
                #    num_iters = 60 
                #else:
                #    num_iters = 10
                num_iters = 100

                #if 'wild' in args.txt_path:     #修正mutation文件数量不足
                #    num_iters = 10 
                #else:
                #    num_iters = 60 
                for i in range(num_iters):
                    coords = []
                    random = np.random.RandomState(i)
                    for key, value in a.items():
                        choice_coord = random.choice(a[key])
                        coords.append(choice_coord)
                    while len(coords) < args.cls_num:
                        #import pdb;pdb.set_trace()
                        coords.append(coords[len(coords)-len(a)])
                    #cut_img(args, path, i, coords)
                    data = pd.DataFrame(coords)
                    #data.to_csv(os.path.join(args.txt_res_path, os.path.basename(path).replace('cls', 'rand_{}'.format(str(i)))), sep='\t', index=0, header=0)
                    data.to_csv(os.path.join(txt_res_path, os.path.basename(path).replace('.txt', '_{}.txt'.format(str(i)))), sep='\t', index=0, header=0)



def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()


