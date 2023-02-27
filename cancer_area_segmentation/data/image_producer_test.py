import os
import sys

import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
from multiprocessing import Pool, Value, Lock

np.random.seed(0)

from torchvision import transforms  # noqa
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

count = Value('i', 0)
lock = Lock()


class ImageDataset(Dataset):

    def __init__(self, data_path, img_size,
                 crop_size=224, normalize=True):
        self._data_path = data_path
        self._img_size = img_size
        self._crop_size = crop_size
        self._normalize = normalize
        # self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self._pre_process()

    def _pre_process(self):
        # find classes
        if sys.version_info >= (3, 5):
            # Faster and available in python 3.5 and above
            classes = [d.name for d in os.scandir(self._data_path) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._data_path) if os.path.isdir(os.path.join(self._data_path, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        # make dataset
        self._items = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(self._data_path, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.split('.')[-1] == 'png':
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        self._items.append(item)

        random.shuffle(self._items)

        self._num_images = len(self._items)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path, label = self._items[idx]
        label = np.array(label, dtype=float)

        return label, path

