import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import glob
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torchsummary import summary
import time

parser = argparse.ArgumentParser(description='MIL2 training')
parser.add_argument('--feat_path', default='/home/sdb/fz/mil_features/60_0923', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--ckpt_path', default=None, metavar='CKPT_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--class_num', default=50, metavar='CLASS_NUM', type=int, help='Clustering Number Class')
parser.add_argument('--log_dir', type=str, default='log9', help='log dir')
parser.add_argument('--epochs',type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                            help='weight decay')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

class Attention(nn.Module):
    def __init__(self, n_class=2):
        super(Attention, self).__init__()
        self.L = 32
        self.D = 16
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear( 64, self.L ),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_class)
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)

        return M, Y_prob

class PatchMethod( torch.utils.data.Dataset ):
    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.raw_samples = glob.glob(os.path.join(root, '*/*.txt'))
        self.unique_samples = list(set(map(lambda x:x[:x.rfind('_')], self.raw_samples)))
        self.samples = []
        self.transform = transform
        for raw_sample in self.unique_samples:
            self.samples.append((raw_sample, int( raw_sample.split( '/' )[-2][0])) )
        if self.mode == 'train':
            random.shuffle( self.samples )

    def __len__(self):
        return len( self.samples )

    def __getitem__(self, index):
        #import pdb;pdb.set_trace()
        path_0, label = self.samples[index]
        path_list = glob.glob(path_0 + '*.txt')
        array = []

        for path_i in path_list:
            with open(path_i, 'r' )as fp:
                json_data = fp.readlines()
                feats = list(map(lambda x:float(x),json_data[0][2:-2].split( ',' )))
            array.append(feats)

        #array = tuple( array )
        #array = torch.stack(torch.from_numpy(array), 0 )
        array = torch.Tensor(array)
        return ( array, label)


def train(epoch, model, train_loader, optimizer, wsi_loss, writer):
    model.train()
    train_loss = 0.
    tpr = 0.
    tnr = 0.
    acc = 0.
    p = 0.
    n = 0.
    for batch_idx, (data, label) in enumerate( train_loader ):
        bag_label = label[0]
        p += 1 - bag_label
        n += bag_label
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        _, y_prob = model.forward( data )
        loss = wsi_loss( y_prob, bag_label.unsqueeze( 0 ) )
        train_loss += loss
        y_prob = F.softmax( y_prob, dim=1 )
        y_p, y_hat = torch.max( y_prob, 1 )
        tpr += (y_hat[0] == bag_label & bag_label == 0)
        tnr += (y_hat[0] == bag_label & bag_label == 1)
        acc += (y_hat[0] == bag_label)
        #print('Batch_idx : {}, loss : {:.3f}, error : {:.3f}, p_error : {:.3f}'
        #      .format( batch_idx, loss.data.cpu(), error,  p_error.data.cpu()[0] ) )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len( train_loader )
    tpr /= p 
    tnr /= n
    acc /= len( train_loader )

    writer.add_scalar( 'data/train_acc', acc, epoch )
    writer.add_scalar( 'data/train_TP_rate', tpr, epoch )
    writer.add_scalar( 'data/train_TN_rate', tnr, epoch )
    writer.add_scalar( 'data/train_loss', train_loss, epoch )

    result_train = '{}, Epoch: {}, Loss: {:.4f}, TP rate: {:.4f}, TN rate: {:.4f} Train accuracy: {:.2f}' \
        .format( time.strftime( "%Y-%m-%d %H:%M:%S" ), epoch, train_loss.data.cpu(), tpr, tnr, acc )

    print( result_train )
    return result_train


def test(epoch, model, test_loader, wsi_loss, writer):
    model.eval()
    test_loss = 0.
    tpr = 0.
    tnr = 0.
    acc = 0.
    p = 0.
    n = 0.
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate( test_loader ):
            bag_label = label[0]
            p += 1- bag_label
            n += bag_label
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            _, y_prob = model.forward( data )
            loss = wsi_loss( y_prob, bag_label.unsqueeze( 0 ) )
            test_loss += loss
            y_prob = F.softmax( y_prob, dim=1 )
            y_p, y_hat = torch.max( y_prob, 1 )
            tpr += (y_hat[0] == bag_label & bag_label == 0)
            tnr += (y_hat[0] == bag_label & bag_label == 1)
            acc += (y_hat[0] == bag_label)
            #print( 'Batch_idx : {}, loss : {:.3f}, error : {:.3f}, p_error : {:.3f}'
            #       .format( batch_idx, loss.data.cpu(), error, p_error.data.cpu()[0] ) )

    test_loss /= len( test_loader )
    tpr /= p 
    tnr /= n 
    acc /= len( test_loader )

    writer.add_scalar( 'data/test_acc', acc, epoch )
    writer.add_scalar( 'data/test_TP_rate', tpr, epoch )
    writer.add_scalar( 'data/test_TN_rate', tnr, epoch )
    writer.add_scalar( 'data/test_loss', test_loss, epoch )
    result_test = '{}, Epoch: {}, Loss: {:.4f}, TP rate: {:.4f}, TN rate: {:.4f}, test accuracy: {:.2f}' \
        .format( time.strftime( "%Y-%m-%d %H:%M:%S" ), epoch, test_loss.data.cpu(), tpr, tnr, acc )
    print( result_test )
    return result_test, acc


def run(args):
    print( 'epoch_{} learning_rate_{}'.format( args.epochs, args.lr ) )
    writer = SummaryWriter( os.path.join( args.log_dir, "epoch" + str( args.epochs ) ) )

    torch.manual_seed( args.seed )
    if args.cuda:
        torch.cuda.manual_seed( args.seed )
        print( '\nGPU is ON!' )

    print( 'Load Train and Test Set' )

    train_dir = os.path.join(args.feat_path, 'train')
    valid_dir = os.path.join( args.feat_path, 'valid' )
    data = PatchMethod( root=train_dir)
    val_data = PatchMethod( root=valid_dir, mode='test')
    train_loader = torch.utils.data.DataLoader( data, shuffle=False, num_workers=0, batch_size=1 )
    valid_loader = torch.utils.data.DataLoader( val_data, shuffle=False, num_workers=0, batch_size=1 )

    save_name_txt = os.path.join(args.log_dir, "train_valid_acc.txt")

    if args.ckpt_path:
        print('Loading Model')
        ckpt = torch.load(args.ckpt_path)
        model = Attention(2)
        model.load_state_dict(ckpt['state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
        start_epoch = ckpt['epoch'] + 1
        model_file = open(save_name_txt, "a")
    else:
        print( 'Init Model' )
        model = Attention( 2 )
        optimizer = optim.Adam( model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg )
        start_epoch = 1
        model_file = open(save_name_txt, "w")

    if args.cuda:
        model = model.cuda()
    #summary( model, (1, 64) )
    wsi_loss = CrossEntropyLoss().cuda()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    cur_test_acc = 0
    for epoch in range(start_epoch, args.epochs + 1):
        print('----------Start Training----------')
        train_result = train(epoch, model, train_loader, optimizer, wsi_loss, writer)
        model_file.write(train_result + '\n')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, os.path.join(args.log_dir, "AMIL_model_latest.ckpt"))
        print('----------Start Testing----------')
        test_result, test_acc = test(epoch, model, valid_loader, wsi_loss, writer)
        model_file.write(test_result + '\n')
        model_file.flush()
        if test_acc > cur_test_acc:
            torch.save( {'epoch': epoch,
                         'state_dict': model.state_dict()},
                        os.path.join(args.log_dir, 'AMIL_model_best.ckpt'))
            cur_test_acc = test_acc
    model_file.close()




if __name__ == '__main__':
    run(args)

