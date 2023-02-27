import sys
import os
import argparse
import logging
import json
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torchvision import models
import torchvision.transforms as transforms
import torch.distributed as dist
from torch import nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from focalloss import FocalLoss

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from data.image_producer import ImageDataset
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cnn_path', default='../configs/cnn_level1.json', metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format')
parser.add_argument('--save_path', default='ckpt/tumor_level0_1_0713', type=str,
                    help='Path to the saved models')
#parser.add_argument('--ckpt_path',default='ckpt/ckpt_2_m_w_0412/best.ckpt', type=str,help='Path to the trained saved models')
parser.add_argument('--ckpt_path',default=None,type=str,help='Path to the trained saved models')
parser.add_argument('--num_workers', default=4, type=int, help='number of'
                    ' workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='2', type=str, help='comma'
                    ' separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                    ' and GPU_1, default 0.')
parser.add_argument('--local_rank', default=-1, type=int,
                            help='node rank for distributed training')
#测试
def chose_model(cnn):
    if cnn['model'] == 'resnet18':
        #model = models.mobilenet_v2(pretrained=False, num_classes=2)
        model = models.shufflenet_v2_x1_0(pretrained=True)
    else:
        raise Exception("I have not add any models. ")
    return model


def train_epoch(args,summary, summary_writer, cnn, model, loss_fn, optimizer, dataloader_train):
    model.train()

    steps = len(dataloader_train)
    batch_size = dataloader_train.batch_size
    #dataiter_train = iter(dataloader_train)
    step = 0
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda", args.local_rank)
    time_now = time.time()
    for data_train,target_train in dataloader_train:
        #data_train, target_train = next(dataiter_train)
        step += 1
        #data_train = Variable(data_train.float().cuda())
        #target_train = Variable(target_train.float().cuda())
        data_train = data_train.to(device)
        target_train = target_train.to(device)
        output = model(data_train)

        loss = loss_fn(output, target_train.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = output.data.max(1, keepdim=True)[1]
        probs = torch.squeeze(probs) # noqa
        # predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)

        #import pdb;pdb.set_trace()
        acc_data = (probs.float() == target_train).type(
            torch.cuda.FloatTensor).sum().data * 1.0 / batch_size
        acc_data0 = ((probs.float() == target_train) & (probs == 0)).type(
            torch.cuda.FloatTensor).sum().data * 1.0 / (target_train == 0).type(torch.cuda.FloatTensor).sum().data
        acc_data1 = ((probs.float() == target_train) & (probs == 1)).type(
            torch.cuda.FloatTensor).sum().data * 1.0 / (target_train == 1).type(torch.cuda.FloatTensor).sum().data
        #acc_data2 = ((probs.float() == target_train) & (probs == 2)).type(
        #    torch.cuda.FloatTensor).sum().data * 1.0 / (target_train == 2).type(torch.cuda.FloatTensor).sum().data
        #acc_data3 = ((probs.float() == target_train) & (probs == 3)).type(
            #torch.cuda.FloatTensor).sum().data * 1.0 / (target_train == 3).type(torch.cuda.FloatTensor).sum().data

        loss_data = loss.data

        time_spent = time.time() - time_now
        logging.info(
            '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
            'Training Acc : {:.3f}, 0-Acc : {:.3f}, 1-Acc : {:.3f}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                summary['step'] + 1, loss_data, acc_data, acc_data0, acc_data1, time_spent))

        summary['step'] += 1

        if summary['step'] % cnn['log_every'] == 0:
            summary_writer.add_scalar('train/loss', loss_data, summary['step'])
            summary_writer.add_scalar('train/acc', acc_data, summary['step'])
            summary_writer.add_scalar('train/acc0', acc_data0, summary['step'])
            summary_writer.add_scalar('train/acc1', acc_data1, summary['step'])
      #      summary_writer.add_scalar('train/acc2', acc_data2, summary['step'])
         #   summary_writer.add_scalar('train/acc3', acc_data3, summary['step'])
        
        #for name, param in model.named_parameters():
        #    summary_writer.add_histogram(name, param.data.cpu().numpy(), global_step=summary['step'])
    summary['epoch'] += 1

    return summary


def valid_epoch(summary, model, loss_fn, dataloader_valid):
    model.eval()

    steps = len(dataloader_valid)
    batch_size = dataloader_valid.batch_size
    dataiter_valid = iter(dataloader_valid)

    loss_sum = 0
    acc_sum = 0
    acc_sum0 = 0
    acc_sum1 = 0
   # acc_sum2 = 0
   # acc_sum3 = 0

    with torch.no_grad():
        for step in range(steps):
            data_valid, target_valid = next(dataiter_valid)
            data_valid = Variable(data_valid.float().cuda())
            target_valid = Variable(target_valid.float().cuda())

            output = model(data_valid)

            loss = loss_fn(output, target_valid.long())

            probs = output.data.max(1, keepdim=True)[1]
            probs = torch.squeeze(probs) # important
            acc_data = (probs.float() == target_valid).type(
                torch.cuda.FloatTensor).sum().data * 1.0 / batch_size
            acc_data0 = ((probs.float() == target_valid) & (probs == 0)).type(
                torch.cuda.FloatTensor).sum().data * 1.0 / (target_valid == 0).type(torch.cuda.FloatTensor).sum().data
            acc_data1 = ((probs.float() == target_valid) & (probs == 1)).type(
                torch.cuda.FloatTensor).sum().data * 1.0 / (target_valid == 1).type(torch.cuda.FloatTensor).sum().data
    #        acc_data2 = ((probs.float() == target_valid) & (probs == 2)).type(
     #           torch.cuda.FloatTensor).sum().data * 1.0 / (target_valid == 2).type(torch.cuda.FloatTensor).sum().data
    #        acc_data3 = ((probs.float() == target_valid) & (probs == 3)).type(
     #           torch.cuda.FloatTensor).sum().data * 1.0 / (target_valid == 3).type(torch.cuda.FloatTensor).sum().data
            loss_data = loss.data

            loss_sum += loss_data
            acc_sum += acc_data
            acc_sum0 += acc_data0
            acc_sum1 += acc_data1
      #      acc_sum2 += acc_data2
      #      acc_sum3 += acc_data3

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps
    summary['acc0'] = acc_sum0 / steps
    summary['acc1'] = acc_sum1 / steps
    #summary['acc2'] = acc_sum2 / steps
    #summary['acc3'] = acc_sum3 / steps

    #for name, param in model.named_parameters():
     #   summary.add_histogram(name, param.data.cpu().numpy(), global_step=summary_train['step'])
    return summary


def run(args):
    with open(args.cnn_path, 'r') as f:
        cnn = json.load(f)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    with open(os.path.join(args.save_path, 'cnn.json'), 'w') as f:
        json.dump(cnn, f, indent=1)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    #num_GPU = len(args.device_ids.split(','))
    num_GPU = 1
    batch_size_train = cnn['batch_size'] * num_GPU
    batch_size_valid = cnn['batch_size'] * num_GPU
    num_workers = args.num_workers

    if args.ckpt_path != None:
        ckpt = torch.load(args.ckpt_path)
        model = chose_model(cnn)
        model.load_state_dict(ckpt['state_dict'])
        #optimizer = optim.load_state_dict(ckpt['optimizer'])
        optimizer = SGD(model.parameters(), lr=cnn['lr'], momentum=cnn['momentum'], )
        start_epoch = ckpt['epoch']
        # model = DataParallel(model, device_ids=None)
        model = nn.parallel.DistributedDataParallel(model.cuda(),device_ids=[args.local_rank],output_device=args.local_rank)
        model = model.cuda()
        loss_fn = CrossEntropyLoss().cuda()
        #loss_fn = FocalLoss().cuda()
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50,80])    
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0, last_epoch=-1)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=14, T_mult=2)
    else:
        model = chose_model(cnn)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, 2) # 须知
        #fc_features = model.classifier[1].in_features
        #model.classifier[1] = nn.Linear(fc_features, 2)
        start_epoch = 0
        # model = DataParallel(model, device_ids=None)
        model = nn.parallel.DistributedDataParallel(model.cuda(),device_ids=[args.local_rank],output_device=args.local_rank)
        model = model.cuda()
        #loss_fn = CrossEntropyLoss().cuda()
        loss_fn = FocalLoss().cuda()
        #weight_p, bias_p = [], []
        #for name, p in model.named_parameters():
        #    if 'bias' in name:
        #        bias_p += [p]
        #    else:
        #        weight_p += [p]
        optimizer = SGD(model.parameters(), lr=cnn['lr'], momentum=cnn['momentum'])
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50,80])    
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=14, T_mult=2)
    # dataset_train = ImageFolder(cnn['data_path_train'])
    # dataset_valid = ImageFolder(cnn['data_path_valid'])


    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    dataset_train = ImageDataset(cnn['data_path_train'],
                                 cnn['image_size'],
                                 trans,
                                 cnn['crop_size'],
                                 cnn['normalize'])
    dataset_valid = ImageDataset(cnn['data_path_valid'],
                                 cnn['image_size'],
                                 trans,
                                 cnn['crop_size'],
                                 cnn['normalize'])

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_valid)

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size_train,
                                  sampler=train_sampler,
                                  pin_memory=False,
                                  num_workers=num_workers,
                                  shuffle=(train_sampler is None))
    # print(len(dataloader_train))
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size=batch_size_valid,
                                  sampler=val_sampler,
                                  pin_memory=False,
                                  num_workers=num_workers,
                                  shuffle=(val_sampler is None))
    # print(len(dataloader_train))

    summary_train = {'epoch': 0 + start_epoch, 'step': 0}
    summary_valid = {'loss': float('inf'), 'acc': 0}
    summary_writer = SummaryWriter(args.save_path)
    loss_valid_best = float('inf')
    f = open(os.path.join(args.save_path,'valid.txt'), 'w')
    old_acc = 0 
    for epoch in range(start_epoch+1, cnn['epoch']):
        #scheduler.load_state_dict()
        train_sampler.set_epoch(epoch)
        summary_train = train_epoch(args,summary_train, summary_writer, cnn, model,
                                    loss_fn, optimizer,
                                    dataloader_train)
        print(scheduler.get_last_lr()[0])
        scheduler.step()
        if dist.get_rank() == 0:
            summary_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], summary_train['epoch'])
        if (summary_train['epoch']+1)%10==0 & dist.get_rank() == 0:
            torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, '{}_train.ckpt'.format(summary_train['epoch'])))

        time_now = time.time()
        summary_valid = valid_epoch(summary_valid, model, loss_fn,
                                    dataloader_valid)
        time_spent = time.time() - time_now

        logging.info('{}, Epoch: {}, step: {}, Validation Loss: {:.5f}, '
                     'Validation ACC: {:.3f}, 0-Acc : {:.3f}, 1-Acc : {:.3f}, Run Time: {:.2f}'
                     .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                             summary_train['step'], summary_valid['loss'],
                             summary_valid['acc'], summary_valid['acc0'], summary_valid['acc1'], time_spent))
        f.write('{}, Epoch: {}, step: {}, Validation Loss: {:.5f}, '
                     'Validation ACC: {:.3f}, 0-Acc : {:.3f}, 1-Acc : {:.3f}, Run Time: {:.2f}'
                     .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                             summary_train['step'], summary_valid['loss'],
                             summary_valid['acc'], summary_valid['acc0'], summary_valid['acc1'], time_spent))
        f.write('\n')
        f.flush()

        
        if dist.get_rank() == 0:
            summary_writer.add_scalar('valid/loss',
                                      summary_valid['loss'], summary_train['step'])
            summary_writer.add_scalar('valid/acc',
                                      summary_valid['acc'], summary_train['step'])
            summary_writer.add_scalar('valid/acc0',
                                      summary_valid['acc0'], summary_train['step'])
            summary_writer.add_scalar('valid/acc1',
                                      summary_valid['acc1'], summary_train['step'])
            for name, param in model.named_parameters():
                summary_writer.add_histogram(name, param.data.cpu().numpy(), global_step=summary_train['epoch'])

        if summary_valid['loss'] < loss_valid_best:
            loss_valid_best = summary_valid['loss']

        if dist.get_rank() == 0 and summary_valid['acc'] > old_acc:
            old_acc = summary_valid['acc']
            torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, 'best.ckpt'))

    f.close()
    summary_writer.close()


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
