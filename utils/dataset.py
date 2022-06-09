# Copyright (C) 2022 ByteDance Inc.
# All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# The software is made available under Creative Commons BY-NC-SA 4.0 license
# by ByteDance Inc. You can use, redistribute, and adapt it
# for non-commercial purposes, as long as you (a) give appropriate credit
# by citing our paper, (b) indicate any changes that you've made,
# and (c) distribute any derivative works under the same license.

# THE AUTHORS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
# DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
# OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from io import BytesIO
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import torch
import cv2
import lmdb
import albumentations
import albumentations.augmentations as A


class MaskDataset(Dataset):
    def __init__(self, path, transform=None, resolution=256, label_size=0, aug=False):

        self.path = path
        self.transform = transform
        self.resolution = resolution
        self.label_size = label_size

        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('image-length'.encode('utf-8')).decode('utf-8'))

        if self.transform is None:
            self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True)
                    ])

        self.aug = aug
        if self.aug == True:
            self.aug_t = albumentations.Compose([
                            A.transforms.HorizontalFlip(p=0.5),
                            A.transforms.ShiftScaleRotate(shift_limit=0.1,
                                                scale_limit=0.2,
                                                rotate_limit=15,
                                                border_mode=cv2.BORDER_CONSTANT,
                                                value=0,
                                                mask_value=0,
                                                p=0.5),
                    ])
        

    def _onehot_mask(self, mask):
        label_size = self.label_size
        labels = np.zeros((label_size, mask.shape[0], mask.shape[1]))
        for i in range(label_size):
            labels[i][mask==i] = 1.0
        
        return labels
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        with self.env.begin(write=False) as txn:
            img = Image.open(BytesIO(txn.get(f'image-{str(idx).zfill(7)}'.encode('utf-8')))).convert('RGB')
            if img.size[0] != self.resolution:
                img = img.resize((self.resolution, self.resolution), resample=Image.LANCZOS)
                
            mask = Image.open(BytesIO(txn.get(f'label-{str(idx).zfill(7)}'.encode('utf-8')))).convert('L')
            if mask.size[0] != self.resolution:
                mask = mask.resize((self.resolution, self.resolution), resample=Image.NEAREST)

        if self.aug:
            augmented = self.aug_t(image=np.array(img), mask=np.array(mask))
            img = Image.fromarray(augmented['image'])
            mask = augmented['mask']
        
        img = self.transform(img)
        mask = self._onehot_mask(np.array(mask))
        mask = torch.tensor(mask, dtype=torch.float) * 2 - 1
        
        return {"image": img, "mask": mask}


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

