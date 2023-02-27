from __future__ import print_function

import numpy as np
# from scipy.misc import imsave
from imageio import imsave
import argparse
import time
import torch.utils.data as data_utils

# from dataloader import MnistBags
from amil_model import Attention

# from __future__ import print_function, division
import os
import glob
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
from torch.nn import CrossEntropyLoss, DataParallel
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from amil_model import Attention, EncAttn
import argparse
import copy
import json

parser = argparse.ArgumentParser(description='Breakthis data_mynet')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--feat_dir', type=str, default='/home/sdb/fz/mil_features/36_0923',
                    help='log dir')                  
parser.add_argument('--ckpt_path', type=str, default='log6/AMIL_model_best_1.ckpt',
                    help='log dir')
parser.add_argument('--log_dir', type=str, default='log2',
                    help='log dir')
parser.add_argument('--feat_flag', type=bool, default=True,
                    help='get features or only test')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


class PatchMethod( torch.utils.data.Dataset ):
    def __init__(self, root='Desktop/screenshots/', transform=None):
        self.root = root
        with open(self.root) as f:
            self.raw_samples = json.load(f)
        self.samples = []
        self.transform = transform
        for raw_sample in self.raw_samples:
            label = int(raw_sample[0].split( '/' )[-2][0])
            for sample in raw_sample:
                self.samples.append((sample, label))


    def __len__(self):
        return len( self.samples )

    def __getitem__(self, index):
        image_dir, label = self.samples[index]
        image_list = glob.glob(os.path.join(image_dir, '*.png'))
        array = []
        for i, image_path in enumerate(image_list ):
            image = Image.open( image_path )
            image = np.array( image )
            array.append( self.transform( image ) )
        array = tuple( array )
        array = torch.stack( array, 0 )
        return (image_dir, array, label)


def test(model, test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    p_error = 0.
    model_file = open(save_name_txt, "w")
    with torch.no_grad():
        for batch_idx, (image_dir, data, label) in enumerate(test_loader):
            bag_label = label[0]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            _, y_prob = model.forward(data)
            loss = wsi_loss( y_prob, bag_label.unsqueeze(0))
            test_loss += loss
            y_prob = F.softmax(y_prob, dim=1)
            y_p, y_hat = torch.max(y_prob,1)
            error = 1. - y_hat.eq( bag_label ).cpu().float().mean().item()
            p_error += 1 - y_p if error==0 else y_p
            test_error += error
            print('Batch_idx : {}, loss : {:.3f}, error : {:.3f}, p_error : {:.3f}'.format(batch_idx, loss.data.cpu(), error, p_error.data.cpu()[0]))
            model_file.write('Batch_idx : {}, loss : {:.3f}, error : {:.3f}, p_error : {:.3f}'
                             .format(batch_idx, loss.data.cpu(), error, p_error.data.cpu()[0]))
            model_file.flush()
    test_error /= len(test_loader)
    p_error /= len(test_loader)
    test_loss /= len(test_loader)
    test_acc = (1 - test_error)*100
    result_test = '{}, Loss: {:.4f}, test error: {:.4f}, p error: {:.4f}, test accuracy: {:.2f}'\
        .format(time.strftime("%Y-%m-%d %H:%M:%S"), test_loss.data.cpu(), test_error, p_error.data.cpu()[0], test_acc)
    print(result_test)
    model_file.write(result_test)
    model_file.flush()
    model_file.close()


def getFeatures(model, test_loader,featdir):
    model.eval()

    with torch.no_grad():
        for batch_idx, (image_dir, data, label) in enumerate(test_loader):
            feat_name = os.path.split(image_dir[0])[-1]
            print('get features: ', feat_name)
            feat_path = os.path.join(featdir, feat_name + '.txt')
            feat_file = open(feat_path, 'w')
            bag_label = label[0]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            features, y_prob = model.forward(data)
            feat_file.write(str(features.cpu()[0].tolist()))
            feat_file.flush()
            feat_file.close()


if __name__ == "__main__":
    torch.manual_seed( args.seed )
    if args.cuda:
        torch.cuda.manual_seed( args.seed )
        print( '\nGPU is ON!' )

    print( 'Load Train and Test Set' )
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    data_path_train_txt = '/home/sdd/zxy/TCGA_data/mil_patch/train_36.json'
    data_path_valid_txt = '/home/sdd/zxy/TCGA_data/mil_patch/valid_36.json'
    data_path_test_txt = '/home/sdd/zxy/TCGA_data/mil_patch/test_36.json'

    normalize = transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
    trans = transforms.Compose( [transforms.ToTensor(), normalize] )
    # trans = transforms.Compose([transforms.ToTensor()])
    data = PatchMethod( root=data_path_train_txt, transform=trans )
    val_data = PatchMethod( root=data_path_valid_txt, transform=trans )
    test_data = PatchMethod( root=data_path_test_txt, transform=trans )

    train_loader = torch.utils.data.DataLoader( data, shuffle=False, num_workers=4, batch_size=1 )
    val_loader = torch.utils.data.DataLoader( val_data, shuffle=False, num_workers=4, batch_size=1 )
    test_loader = torch.utils.data.DataLoader( test_data, shuffle=False, num_workers=4, batch_size=1 )
    if args.feat_flag:
        if not os.path.exists( args.feat_dir ):
            os.makedirs( args.feat_dir )
    else:
        save_name_txt = os.path.join(args.log_dir, "valid_acc.txt")
        wsi_loss = CrossEntropyLoss().cuda()
        if not os.path.exists( args.log_dir ):
            os.makedirs( args.log_dir )

    print('Loading Model')
    ckpt = torch.load(args.ckpt_path)
    model = Attention(2)
    model.load_state_dict(ckpt['state_dict'])

    if args.cuda:
        model = model.cuda()
    summary( model, (3, 512, 512) )

    if args.feat_flag:
        print( '----------Geting features----------' )
        getFeatures(model, train_loader, os.path.join(args.feat_dir, 'train'))
        getFeatures(model, val_loader, os.path.join(args.feat_dir, 'valid'))
        getFeatures(model, test_loader, os.path.join(args.feat_dir, 'test'))
    else:
        print( '----------Start Testing----------' )
        test(model, test_loader, save_name_txt)
