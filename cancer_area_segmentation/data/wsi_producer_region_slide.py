import numpy as np
import openslide
from torch.utils.data import DataLoader
import math

np.random.seed(0)


class WSIPatchDataset(object):
    def __init__(self, wsi_path,mask_path, level, image_size=256, crop_size=224,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._level = level
        self._image_size = image_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        self._mask = np.load(self._mask_path)
        #import pdb;pdb.set_trace()
        self._slide = openslide.OpenSlide(self._wsi_path)
        self.X_slide, self.Y_slide = self._slide.level_dimensions[0]
        X_mask, Y_mask = self._mask.shape
        if self.X_slide // X_mask != self.Y_slide // Y_mask:
            raise Exception('Slide/Mask dimension does not match ,'
                            'X_slide / X_mask: {} / {}, '
                            'Y_slide / Y_mask: {} / {}'
                            .format(self.X_slide, X_mask, self.Y_slide, Y_mask))
        self._resolution = self.X_slide // X_mask
        self._X_idcs, self._Y_idcs = np.where(self._mask)
        self._idcs_num = len(self._X_idcs)
        #self.rate = self._image_size // resolution
        #self._coords = []
        #rate = 2**self._level
        #for x in range(0, self.X_slide, self._image_size*rate):
        #    temp = x//resolution
        #    for y in range(0, self.Y_slide, self._image_size*rate):
        #        if(self._mask[temp, y//resolution]):
        #            self._coords.append((x, y))



    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx] 
        x_center = int((x_mask + 0.5) * self._resolution)
        y_center = int((y_mask + 0.5) * self._resolution)
        x = int(x_center - self._image_size / 2)
        y = int(y_center - self._image_size / 2)
        #x_idx = self._coords[coord_idx][0]
        #y_idx = self._coords[coord_idx][1]

        img = self._slide.read_region(
            (x, y), 1, (self._image_size, self._image_size)).convert('RGB')

        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        return (img, x_mask, y_mask)


# dataloader = DataLoader(
#     WSIPatchDataset(r'D:\publicData\fuzhong\svs\mutation\TCGA-BT-A20T-01Z-00-DX1.96460E53-65E0-425F-B079-939D7AA537BE.svs',
#                     r'D:\publicData\fuzhong\tissueNpy\mutation\TCGA-BT-A20T-01Z-00-DX1.96460E53-65E0-425F-B079-939D7AA537BE_tissue.npy',
#                     level=1, image_size=768,
#                     crop_size=672, normalize=True),
#                     batch_size=128, drop_last=False)
# print(len(dataloader))
# for (data, x_mask, y_mask) in dataloader:
#     print(x_mask,y_mask)
