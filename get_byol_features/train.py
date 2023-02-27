import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL
import pytorch_lightning as pl

import glob

# test model, a resnet 50

resnet = models.resnet50(pretrained=True)

# arguments

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', default='/home/sdd/zxy/TCGA_data/mil_patch', type=str,
                       help='path to your folder of images for self-supervised learning')

args = parser.parse_args()

# constants

BATCH_SIZE = 64 
EPOCHS     = 1000
LR         = 3e-4
NUM_GPUS   = 4
IMAGE_SIZE = 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
# NUM_WORKERS = multiprocessing.cpu_count()
NUM_WORKERS = 4
# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path_dir in glob.glob(os.path.join(folder, '*')):
            ext = path_dir.split('/')[-1]
            if ext == '0_mutation':
                for i in range(6):
                    path_i = glob.glob(os.path.join(path_dir, '*_{}'.format(str(i))))
                    for path_i_i in path_i:
                        self.paths += glob.glob(os.path.join(path_i_i, '*.png'))
            elif ext == '1_wild':
                path_i = glob.glob(os.path.join(path_dir, '*_0'))
                for path_i_i in path_i:
                    self.paths += glob.glob(os.path.join(path_i_i, '*.png'))
            else:
                raise ValueError('Invalid class name: %s' % ext)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

# main

if __name__ == '__main__':
    train_image_dir = os.path.join(args.image_folder, 'train')
    valid_image_dir = os.path.join(args.image_folder, 'valid')
    train_ds = ImagesDataset(train_image_dir, IMAGE_SIZE)
    val_ds = ImagesDataset(valid_image_dir, IMAGE_SIZE )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = SelfSupervisedLearner(
        resnet,
        image_size=IMAGE_SIZE,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    )

    trainer = pl.Trainer(
        gpus=NUM_GPUS,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True
    )

    #train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
