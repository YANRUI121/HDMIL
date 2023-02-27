import sys
import os
import argparse
import logging
import json
import time
import glob
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from torch.nn import DataParallel

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from cancer_area_segmentation.data.wsi_producer_slide_window import WSIPatchDataset


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
#parser.add_argument('--wsi_path', default='/home/sdc/pangguang/svs/train/mutation', metavar='WSI_PATH', type=str,
#                    help='Path to the input WSI file')
parser.add_argument('--wsi_path', default='/home/sdc/fuzhong-linchuang/svs/test/wild', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--ckpt_path', default='ckpt/tumor_level1_linchuang/best.ckpt', metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('--model_name', default='MobileNetV2', metavar='MODEL_NAME', type=str,
                           help='name of the model')
#parser.add_argument('--mask_path', default='/home/sdd/zxy/TCGA_data/npy_tissue/mutation', metavar='MASK_PATH', type=str,
#                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('--mask_path', default='/home/sdc/fuzhong-linchuang/midFiles/tissue/wild', metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('--probs_map_path', default='/home/sdc/fuzhong-linchuang/midFiles/probNpy_linchuang_0819/wild', metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='3', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=4, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--level', default=1, type=int, help='level of wsi')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--patch_size', default=256, type=int, help='the size of patch')


def chose_model(mod):
    if mod == 'MobileNetV2':
        #model = models.resnet18(pretrained=False, num_classes=2)
        model = models.mobilenet_v2(pretrained=False, num_classes=2)
    else:
        raise Exception("I have not add any models. ")
    return model


def get_probs_map(model, dataloader):
    rate = dataloader.dataset.rate
    print(rate)
    print(dataloader.dataset.x_mask,dataloader.dataset.y_mask)
    probs_map = np.zeros((math.ceil(dataloader.dataset.x_mask/rate), math.ceil(dataloader.dataset.y_mask/rate)))
    #probs_map = np.zeros((math.ceil(dataloader.dataset.x_mask), math.ceil(dataloader.dataset.y_mask)))
    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    with torch.no_grad():
        #import pdb;pdb.set_trace()
        for (data, x_mask, y_mask) in dataloader:
            #data = Variable(data.cuda(async=True))
            output = model(data)
            # because of torch.squeeze at the end of forward in resnet.py, if the
            # len of dim_0 (batch_size) of data is 1, then output removes this dim.
            # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
            #import pdb;pdb.set_trace()
            probRes = output.cpu()
            probRes = torch.squeeze(probRes,1)
            if len(probRes.shape) <= 1:
                probs = probRes.sigmoid().cpu().data.numpy().flatten()
                probs_map[x_mask, y_mask] = probs
            else:
                prob_cls = probRes.softmax(1)
                probs = torch.argmax(probRes, dim=1)
                #probs = probRes[:,0].sigmoid().cpu().data.numpy().flatten()
                probs_map[x_mask, y_mask] = prob_cls[:,1]
            count += 1

            time_spent = time.time() - time_now
            time_now = time.time()
            logging.info(
            '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent))
    #torch.cuda.empty_cache()
    return probs_map


def make_dataloader(args, mask_path, flip='NONE', rotate='NONE'):
    num_GPU = len(args.GPU.split(','))
    num_workers = args.num_workers

    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    dataloader = DataLoader(
        WSIPatchDataset(args.wsi_path, mask_path, trans, args.level,
                        image_size=args.patch_size,
                        step_size=int(args.patch_size/4), normalize=True,
                        flip=flip, rotate=rotate),
        batch_size=args.batch_size*num_GPU, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    paths = glob.glob(os.path.join(args.wsi_path, '*.svs'))
    paths = paths[::-1]
    for path in paths:
        print(path)
        args.wsi_path = path
        npy_name = os.path.basename(path)
        npy_path = os.path.join(args.probs_map_path, npy_name[:-4] + '_prob.npy')
        #if os.path.exists(npy_path):
        #    continue
        ckpt = torch.load(args.ckpt_path)
        #mask_path = os.path.join(args.mask_path,npy_name[:-4] + '_prob.npy')
        mask_path = os.path.join(args.mask_path,npy_name[:-4] + '.npy')
        #mask = np.load(mask_path)
        model = chose_model(args.model_name)
        model.load_state_dict(ckpt['state_dict'])
        model = DataParallel(model, device_ids=None)
        model = model.cuda().eval()
        start_time = time.time()
        dataloader = make_dataloader(
            args, mask_path, flip='NONE', rotate='NONE')
        probs_map = get_probs_map(model, dataloader)
        print(path)
        print('one wsi time:', time.time()-start_time)
        np.save(npy_path, probs_map)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
