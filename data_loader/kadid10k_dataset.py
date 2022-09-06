import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomCrop, Normalize, RandomRotation
import torchvision.transforms.functional as F

import cv2
import csv
import os
import numpy as np

class Kadid10kDataset(Dataset):
    def __init__(self, width, height, img_path, flist, crop_size, is_train=True):
        self.width = width
        self.height = height
        self.img_path = img_path
        self.flist = flist
        self.crop_size = crop_size
        self.is_train = is_train
        self.data = self._load_flist(flist)

    def __len__(self):
        return len(self.data['mos'])

    def __getitem__(self, idx):
        dist_path = os.path.join(self.img_path, self.data['dist_path'][idx])
        ref_path = os.path.join(self.img_path, self.data['ref_path'][idx])
        img_dist = self._load_image(dist_path)
        img_ref = self._load_image(ref_path)
        if self.is_train:
            img_dist = self._to_tensor(img_dist)
            img_ref = self._to_tensor(img_ref)
            img_dist, img_ref = self._random_rotate(img_dist, img_ref)
            img_dist, img_ref = self._horizontal_flip(img_dist, img_ref)
            img_dist, img_ref = self._vertical_flip(img_dist, img_ref)
            img_dist, img_ref = self._random_crop(img_dist, img_ref, crop_size=self.crop_size)
            img_dist = self._normalize(img_dist)
            img_ref = self._normalize(img_ref)

        mos = self.data['mos'][idx]
        mos_norm = self.data['mos_norm'][idx]
        return img_ref, img_dist, mos, mos_norm

    def _load_flist(self, flist):
        data = {'dist_path': [], 'ref_path': [], 'mos': [], 'mos_norm': []}
        with open(flist) as f:
            reader = csv.reader(f)
            _ = next(reader)
            for idx, row in enumerate(reader):
                data['dist_path'].append(row[0])
                data['ref_path'].append(row[1])
                data['mos'].append(float(row[2]))
        assert len(data['mos']) == len(data['dist_path'])
        data['mos_norm'] = self.mosMinMaxNorm(data['mos'])
        return data

    def mosMinMaxNorm(self, mos):
        mos_norm = []
        mos_max = max(mos)
        mos_min = min(mos)
        for idx in range(len(mos)):
            mos_norm.append(float(1. - (mos[idx] - mos_min) / (mos_max - mos_min)))
        return mos_norm

    def _load_image(self, data):
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _to_tensor(self, x):
        x = ToTensor()(x).float()
        return x

    def _normalize(self, x):
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = normalize(x)
        return x

    def _horizontal_flip(self, x, y):
        if torch.rand(1) < 0.5:
            return F.hflip(x), F.hflip(y)
        return x, y

    def _vertical_flip(self, x, y):
        if torch.rand(1) < 0.5:
            return F.vflip(x), F.vflip(y)
        return x, y

    def _random_rotate(self, x, y):
        degrees = [45, 90]
        if torch.rand(1) < 0.5:
            idx = np.random.randint(0, 2)
            x = F.rotate(x, degrees[idx])
            y = F.rotate(y, degrees[idx])
        return x, y

    def _random_crop(self, x, y, crop_size):
        randomcrop = RandomCrop(crop_size)
        i, j, h, w = randomcrop.get_params(x, output_size=(crop_size, crop_size))
        x = F.crop(x, i, j, h, w)
        y = F.crop(y, i, j, h, w)
        return x, y