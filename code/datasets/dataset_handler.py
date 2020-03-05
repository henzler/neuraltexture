import importlib
import torch
import numpy as np
import random


class DatasetHandler(object):

    def __init__(self, param):

        self.param = param
        self.dataloader_train, self.dataset_train = self.setup_dataset('train')
        self.dataloader_val, self.dataset_val = self.setup_dataset('val')
        self.dataloader_test, self.dataset_test = self.setup_dataset('test')

    def setup_dataset(self, type):

        try:
            dataset_module = importlib.import_module(self.param.dataset.name)
            dataset = getattr(dataset_module, 'CustomDataset')(self.param, type)
        except Exception:
            raise Exception('{} is not configured properly'.format(self.param.dataset.name))

        shuffle = False

        if type == 'train':
            shuffle = True

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.param.train.bs, shuffle=shuffle, num_workers=self.param.n_workers, drop_last=True)

        return dataloader, dataset
