from __future__ import print_function

import numpy as np
# from scipy.misc import imsave
from imageio import imsave
import argparse
import time
import math
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
import torch.optim.lr_scheduler as lr_scheduler
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from amil_model import Attention, EncAttn
from tensorboardX import SummaryWriter
import argparse
import copy
import json

parser = argparse.ArgumentParser(description='Breakthis data_mynet')
parser.add_argument('--epochs',type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--log_dir', type=str, default='log1',
                    help='log dir')                  
# parser.add_argument('--ckpt_path', type=str, default='log2/AMIL_model_latest.ckpt',
#                     help='log dir')
parser.add_argument('--ckpt_path', type=str, default=None,
                   help='log dir')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



class PatchMethod( torch.utils.data.Dataset ):
    def __init__(self, root='Desktop/screenshots/', mode='train', transform=None):
        self.root = root
        self.mode = mode
        with open(self.root) as f:
            self.raw_samples = json.load(f)
        self.samples = []
        self.transform = transform
        for raw_sample in self.raw_samples:
            label = int(raw_sample[0].split( '/' )[-2][0])
            for sample in raw_sample:
                self.samples.append((sample, label))
        if self.mode == 'train':
            random.shuffle( self.samples )

    def __len__(self):
        return len( self.samples )

    def __getitem__(self, index):
        image_dir, label = self.samples[index]
        image_list = glob.glob(os.path.join(image_dir, '*.png'))
        #print(image_dir)
        array = []
        for i, image_path in enumerate(image_list ):
            image = Image.open( image_path )
            image = np.array( image )
            array.append( self.transform( image ) )
        #import pdb;pdb.set_trace()
        array = tuple( array )
        array = torch.stack( array, 0 )
        return (image_dir, array, label)


def train(epoch, model, train_loader, optimizer, writer):
    model.train()
    train_loss = 0.
    tpr = 0.
    tnr = 0.
    acc = 0.
    p = 0.
    n = 0.
    train_length = len(train_loader)
    for batch_idx, (dir_b, data, label) in enumerate(train_loader):
        #print(dir_b)
        bag_label = label[0]
        p += 1 - bag_label
        n += bag_label
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()

        _, y_prob = model.forward(data)
        loss = wsi_loss(y_prob, bag_label.unsqueeze(0))
        train_loss += loss
        y_prob = F.softmax(y_prob, dim=1)
        y_p, y_hat = torch.max(y_prob, 1)
        tpr += (y_hat[0] == bag_label & bag_label == 0)
        tnr += (y_hat[0] == bag_label & bag_label == 1)
        acc += (y_hat[0] == bag_label)
        print('Batch_idx : {}/{}, loss : {:.3f}, tpr : {}/{}, tnr : {}/{}, acc : {}/{}'
              .format(batch_idx, train_length, loss.data.cpu(), tpr, p, tnr, n, acc, p+n))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= train_length
    tpr /= p
    tnr /= n
    acc /= train_length

    writer.add_scalar( 'data/train_acc', acc, epoch )
    writer.add_scalar( 'data/train_TP_rate', tpr, epoch )
    writer.add_scalar( 'data/train_TN_rate', tnr, epoch )
    writer.add_scalar( 'data/train_loss', train_loss, epoch )

    result_train = '{}, Epoch: {}, Loss: {:.4f}, TP rate: {:.4f}, TN rate: {:.4f} Train accuracy: {:.2f}' \
        .format( time.strftime( "%Y-%m-%d %H:%M:%S" ), epoch, train_loss.data.cpu(), tpr, tnr, acc )

    print(result_train)
    return result_train

def test(epoch, model, test_loader, writer):
    model.eval()
    test_loss = 0.
    tpr = 0.
    tnr = 0.
    acc = 0.
    p = 0.
    n = 0.
    test_length = len(test_loader)
    with torch.no_grad():
        for batch_idx,  (_, data, label) in enumerate(test_loader):
            bag_label = label[0]
            p += 1 - bag_label
            n += bag_label
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            _, y_prob = model.forward(data)
            loss = wsi_loss( y_prob, bag_label.unsqueeze(0))
            test_loss += loss
            y_prob = F.softmax(y_prob, dim=1)
            y_p, y_hat = torch.max(y_prob,1)
            tpr += (y_hat[0] == bag_label & bag_label == 0)
            tnr += (y_hat[0] == bag_label & bag_label == 1)
            acc += (y_hat[0] == bag_label)
            print( 'Batch_idx : {}/{}, loss : {:.3f}, tpr : {}/{}, tnr : {}/{}, acc : {}/{}'
                   .format( batch_idx, test_length, loss.data.cpu(), tpr, p, tnr, n, acc, p+n ) )

    test_loss /= test_length
    tpr /= p
    tnr /= n
    acc /= test_length

    writer.add_scalar( 'data/test_acc', acc, epoch )
    writer.add_scalar( 'data/test_TP_rate', tpr, epoch )
    writer.add_scalar( 'data/test_TN_rate', tnr, epoch )
    writer.add_scalar( 'data/test_loss', test_loss, epoch )
    result_test = '{}, Epoch: {}, Loss: {:.4f}, TP rate: {:.4f}, TN rate: {:.4f}, test accuracy: {:.2f}' \
        .format( time.strftime( "%Y-%m-%d %H:%M:%S" ), epoch, test_loss.data.cpu(), tpr, tnr, acc )
    print(result_test)
    return result_test, tpr, tnr


if __name__ == "__main__":
    print( 'epoch_{} learning_rate_{}'.format( args.epochs, args.lr ) )
    writer = SummaryWriter( os.path.join( args.log_dir, "epoch" + str( args.epochs ) ) )

    torch.manual_seed( args.seed )
    if args.cuda:
        torch.cuda.manual_seed( args.seed )
        print( '\nGPU is ON!' )

    print( 'Load Train and Test Set' )
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    data_path_train_txt = '/home/sdd/zxy/TCGA_data/all_mil_patch/FGFR3/train_m36_w6.json'
    data_path_test_txt = '/home/sdd/zxy/TCGA_data/all_mil_patch/FGFR3/valid_m36_w6.json'

    normalize = transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
    trans = transforms.Compose( [transforms.ToTensor(), normalize] )
    # trans = transforms.Compose([transforms.ToTensor()])
    data = PatchMethod( root=data_path_train_txt, transform=trans )
    val_data = PatchMethod( root=data_path_test_txt, mode='test', transform=trans )

    train_loader = torch.utils.data.DataLoader( data, shuffle=False, num_workers=4, batch_size=1 )
    test_loader = torch.utils.data.DataLoader( val_data, shuffle=False, num_workers=4, batch_size=1 )

    save_name_txt = os.path.join(args.log_dir, "train_valid_acc.txt")
    if args.ckpt_path:
        print('Loading Model')
        ckpt = torch.load(args.ckpt_path)
        model = Attention(2)
        model.load_state_dict(ckpt['state_dict'])
        #model.load_state_dict(ckpt)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
        start_epoch = ckpt['epoch'] + 1
        #start_epoch = 15
        model_file = open(save_name_txt, "a")
    else:
        print( 'Init Model' )
        # model = Attention(2, bn_track_running_stats=True)
        model = Attention( 2 )
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lr) + args.lr  # cosine
        optimizer = optim.Adam( model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg )
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        start_epoch = 1
        model_file = open(save_name_txt, "w")
        # model = DataParallel(model, device_ids=None)
        # model = model.cuda()
    if args.cuda:
        model = model.cuda()
    summary( model, (3, 512, 512) )
    wsi_loss = CrossEntropyLoss().cuda()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    cur_test_acc = 0
    for epoch in range(start_epoch, args.epochs + 1):
        print('----------Start Training----------')
        train_result = train(epoch, model, train_loader, optimizer, writer)
        scheduler.step()
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, os.path.join(args.log_dir, "AMIL_model_latest.ckpt"))
        print('----------Start Testing----------')
        test_result, tpr, tnr = test(epoch, model, test_loader, writer)
        model_file.write(test_result + '\n')
        model_file.write(train_result + '\n')
        model_file.flush()
        if tpr + tnr > cur_test_acc:
            torch.save( {'epoch': epoch,
                         'state_dict': model.state_dict()},
                        os.path.join(args.log_dir, 'AMIL_model_best.ckpt'))
            cur_test_acc = tpr + tnr
    model_file.close()
