import sys
import os
import argparse
import logging
import json
import time
import glob
import math
import openslide
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
#from torchvision import models
import myResnet
# import myMobileNetV2
from torch import nn
from torch.nn import DataParallel
import json
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from cancer_area_segmentation.data.wsi_producer import WSIPatchDataset


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('--wsi_path', default='/home/sdc/pangguang/svs/train/wild', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--ckpt_path', default='ckpt_log/ckpt_2_FGFR3_1/best_12_7.ckpt', metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('--model_name', default='resnet18', metavar='MODEL_NAME', type=str, help='name of the model')
parser.add_argument('--mask_path', default='/home/sdd/zxy/TCGA_data/all_prob_npy/tumor_npy', metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('--feat_path', default='/home/sdd/zxy/TCGA_data/all_feat_txt/FGFR3', metavar='FEAT_PATH', type=str,
                    help='Path to the feature txt')
parser.add_argument('--probs_map_path', default='/home/sdd/zxy/TCGA_data/all_prob_npy/FGFR3', metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='5', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=4, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--level', default=1, type=int, help='level of wsi')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--patch_size', default=512, type=int, help='the size of patch')
parser.add_argument('--step', default=256, type=int, help='the size of patch')

def chose_model(mod):
    if mod == 'resnet18':
        model = myResnet.resnet18(pretrained=False, num_classes=2)
    else:
        raise Exception("I have not add any models. ")
    return model


def get_probs_map(args, model, dataloader, path):
    rate = dataloader.dataset.rate
    print(rate)
    print(dataloader.dataset.x_mask,dataloader.dataset.y_mask)
    probs_map = np.zeros((math.ceil(dataloader.dataset.x_mask/rate), math.ceil(dataloader.dataset.y_mask/rate)))
    #probs_map = np.zeros((dataloader.dataset.X_idces, dataloader.dataset.Y_idces))
    num_batch = len(dataloader)
    count = 0
    time_now = time.time()
    json_name = os.path.basename(path)[:-4]
    json_path = os.path.join(args.feat_path,json_name + '.txt')
    with torch.no_grad():
        with open(json_path, "w") as f:
            for (data, x_mask, y_mask) in dataloader:
                #data = Variable(data.cuda(async=True))
                output = model(data)
                # because of torch.squeeze at the end of forward in resnet.py, if the
                # len of dim_0 (batch_size) of data is 1, then output removes this dim.
                # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
                #import pdb;pdb.set_trace()
                probRes = output[0].cpu()
                probRes = torch.squeeze(probRes,1)
                if len(probRes.shape) <= 1:
                    probs = probRes.sigmoid().cpu().data.numpy().flatten()
                    probs_map[x_mask, y_mask] = probs
                else:
                    #import pdb;pdb.set_trace()
                    prob_cls = probRes.softmax(1)
                    probs = torch.argmax(probRes, dim=1)
                    #probs = probRes[:,0].sigmoid().cpu().data.numpy().flatten()
                    #for p in range(data.shape[0]):
                    #    if torch.mean(data[p])>0.5:
                    #        prob_cls[p,0]=0
                    #        probs[p]=1
                    probs_map[x_mask, y_mask] = prob_cls[:,0].cpu()
                #idList = list(zip(x_mask[probs>0.5].tolist(), y_mask[probs>0.5].tolist()))
                #featureList = list(output[1][probs>0.5].tolist())
                idList = list(zip(x_mask.tolist(), y_mask.tolist()))
                featureList = list(output[1].tolist())
                for idNum,idName in enumerate(idList):
                    #import pdb;pdb.set_trace()
                    featureDict = {}
                    featureDict['{}'.format(str(idName))]=featureList[idNum]
                    json.dump(featureDict, f)
                    f.write('\n')
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
    batch_size = args.batch_size * num_GPU
    num_workers = args.num_workers

    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    dataloader = DataLoader(
        WSIPatchDataset(args.wsi_path, mask_path, trans, args.level,
                        image_size=args.patch_size,
                        step_size=args.step, normalize=True,
                        flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    ff = os.walk(args.wsi_path)
    paths = []
    for root, dirs, files in ff:
        for file in files:
            if os.path.splitext(file)[1] == '.svs':
                paths.append(os.path.join(root, file))
    #paths = paths[::-1]
    for path in paths:
        print(path)
        #slide = openslide.open_slide(path)
        #if slide.level_downsamples[1] == 4:
        #    continue
        args.wsi_path = path
        npy_name = os.path.basename(path)
        npy_path = os.path.join(args.probs_map_path, npy_name[:-4] + '_prob.npy')
        if os.path.exists(npy_path):
           continue
        mask_path = os.path.join(args.mask_path,npy_name[:-4] + '_prob.npy')  #Generating heat maps based on cancer regions
        #mask_path = os.path.join(args.mask_path,npy_name[:-4] + '.npy')
        # mask = np.load(mask_path)
        ckpt = torch.load(args.ckpt_path)
        model = chose_model(args.model_name)
        #import pdb;pdb.set_trace()
        model.load_state_dict(ckpt['state_dict'])
        #print(ckpt['state_dict'])
        model = DataParallel(model, device_ids=None)
        model = model.cuda().eval()
        dataloader = make_dataloader(
            args, mask_path, flip='NONE', rotate='NONE')
        probs_map = get_probs_map(args, model, dataloader, path)

        np.save(npy_path, probs_map)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
