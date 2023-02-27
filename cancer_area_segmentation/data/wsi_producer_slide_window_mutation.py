import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL
import math
from normalizeStaining import normalizeStaining

class WSIPatchDataset(Dataset):

    def __init__(self, wsi_path, mask_path, trans, level, image_size=256, step_size=256,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._image_size = image_size
        self._step = step_size
        self._trans = trans
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._level = level
        self._pre_process()

    def _pre_process(self):
        self._mask = np.load(self._mask_path)
        #import pdb;pdb.set_trace()
        self._slide = openslide.OpenSlide(self._wsi_path)
        self.down_samples = int(self._slide.level_downsamples[self._level])
        self.X_slide_0, self.Y_slide_0 = self._slide.level_dimensions[0]
        self.X_slide, self.Y_slide = self._slide.level_dimensions[self._level]
        self.x_mask, self.y_mask = self._mask.shape
        self._resolution = round(self.X_slide / self.x_mask)
        print(self._slide.level_dimensions[0], self.X_slide, self.x_mask, self._resolution)
        #if abs(self._resolution - 256) > 2:
        #    raise Exception('Resolution (X_slide / X_mask) is not power of 2: '
        #                '{}'.format(self._resolution))
        #self._resolution = 256
        if not np.log2(self._resolution).is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2: '
                            '{}'.format(self._resolution))
        #if self._resolution > self._image_size:
        #    raise Exception('Resolution {} is greater than {}'.format(self._resolution,self._image_size))
        #self.rate = self._image_size / self._resolution
        #self.rate = self._step / self._resolution
        self.rate = 1 
        self.x_cors, self.y_cors = list(np.where(self._mask>0.5))
        #self.x_cors, self.y_cors = list(map(lambda x: x//self.rate, np.where(self._mask > 0.5)))
        #self.x_cors, self.y_cors = list(map(lambda x: x, np.where(self._mask > 0.5)))
        #self.x_cors, self.y_cors = list(map(lambda x: x, np.where(self._mask)))
        self.coords = list(set(zip(self.x_cors,self.y_cors)))
       
        #self.X_idces = math.ceil((self.X_slide / self._image_size)/self.down_samples)
        #self.X_idces = math.ceil((self.X_slide / self._image_size))
        #self.Y_idces = math.ceil((self.Y_slide / self._image_size))
        #self.Y_idces = math.ceil((self.Y_slide / self._image_size)/self.down_samples)

        # all the idces for tissue region from the tissue mask

    def __len__(self):
        return len(self.coords) 

    def __getitem__(self, coord_idx):
        #import pdb;pdb.set_trace()
        #x_idx = coord_idx // self.Y_idces
        #y_idx = coord_idx % self.Y_idces
        #x_mask = int(self._image_size * x_idx * self.down_samples)
        #x_mask = int(self._image_size * x_idx)
        #y_mask = int(self._image_size * y_idx)
        #y_mask = int(self._image_size * y_idx * self.down_samples)
        #import pdb;pdb.set_trace()
        #x_mask = int(self.coords[coord_idx][0])*self._resolution
        #y_mask = int(self.coords[coord_idx][1])*self._resolution
        x_mask = int(self.coords[coord_idx][0])*self._step
        y_mask = int(self.coords[coord_idx][1])*self._step
        #print(x_mask,y_mask)
        #x = x_mask if int(x_mask + self._image_size * self.down_samples) < self.X_slide else int(self.X_slide - self._image_size * self.down_samples)
        #y = y_mask if int(y_mask + self._image_size * self.down_samples) < self.Y_slide else int(self.Y_slide - self._image_size * self.down_samples)
        x = x_mask if int(x_mask + self._image_size * self.down_samples) < self.X_slide_0 else int(self.X_slide_0 - self._image_size * self.down_samples)
        y = y_mask if int(y_mask + self._image_size * self.down_samples) < self.Y_slide_0 else int(self.Y_slide_0 - self._image_size * self.down_samples)

        #print(x,y)
        #print(x,y,self._image_size)
        #img = self._slide.read_region((x, y), self._level, (self._image_size, self._image_size)).convert('RGB')
        if self.down_samples != 4:
            img1 = self._slide.read_region(
            (x, y), 0, (self._image_size*4, self._image_size*4)).convert('RGB')
            img = img1.resize((self._image_size, self._image_size))
        else:
            img = self._slide.read_region((x, y), 1, (self._image_size, self._image_size)).convert('RGB')
        #if self._flip == 'FLIP_LEFT_RIGHT':
        #    img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        #if self._rotate == 'ROTATE_90':
        #    img = img.transpose(PIL.Image.ROTATE_90)

        #if self._rotate == 'ROTATE_180':
        #    img = img.transpose(PIL.Image.ROTATE_180)

        ##if self._rotate == 'ROTATE_270':
        #    img = img.transpose(PIL.Image.ROTATE_270)

            # PIL image:   H x W x C
            # torch image: C X H X W
        #img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        #if self._normalize:
        #    img = (img - 128.0) / 128.0
        #img = normalizeStaining(np.array(img))
        img = self._trans(img)

        return (img,int(self.coords[coord_idx][0]),int(self.coords[coord_idx][1]))

# dataloader = DataLoader(
#     WSIPatchDataset(r'D:\publicData\TCGA\svs\TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.svs',
#                     image_size=768,
#                     crop_size=672, normalize=True),
#                     batch_size=128, drop_last=False)
# num_batch = len(dataloader)
# for (data, x_mask, y_mask) in dataloader:
#     print(x_mask,y_mask)
