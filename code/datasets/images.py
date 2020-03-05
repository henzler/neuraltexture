from torch.utils.data import Dataset
import os
import json
import csv
import torch
import utils.io as io
import utils.utils as util
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
import re


class CustomDataset(Dataset):

    def __init__(self, param, dataset_mode='train'):

        self.param = param
        self.device = param.device
        self.path_dir = Path(param.dataset.path)
        self.dataset_mode = dataset_mode
        self.image_res = param.image.image_res
        self.use_single = param.dataset.use_single

        self.tmp_image = None
        self.samples = self.get_samples()
        self.dataset_length = 1 if self.use_single != -1 and self.dataset_mode == 'test' else len(self.samples)

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(self.image_res),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()])

    def __len__(self):
        return self.dataset_length

    def get_samples(self):

        samples = []
        types = ('*.jpeg', '*.jpg', '*.png')

        split = 'test' if self.dataset_mode == 'val' else self.dataset_mode

        if self.use_single != -1 or self.dataset_mode == 'test':
            for type in types:
                samples += list(Path(self.path_dir).glob('**/{}'.format(type)))
        else:
            for type in types:
                samples += list(Path(self.path_dir / split).glob('**/{}'.format(type)))

        samples.sort()

        if self.dataset_mode == 'train':
            samples = samples * self.param.train.bs * 1000  # 1000 can be removed if data set is big
        else:
            samples = samples * self.param.train.bs

        return samples

    def __getitem__(self, idx):

        idx = idx if self.use_single == -1 else self.use_single

        filepath = self.samples[idx]

        if self.tmp_image is not None:
            file = self.tmp_image
        else:
            file = io.load_image(str(Path(filepath)))
            file = util.numpy_to_pytorch(file)
            if self.use_single != -1:
                self.tmp_image = file

        if self.use_single == -1:
            file = self.transforms(file)
        else:
            file = file[:, :self.image_res, :self.image_res]

        if self.dataset_mode == 'test':
            return file, Path(filepath).stem

        return file
