import torch
import importlib
from attrdict import AttrDict
import numpy as np
import random
from datasets.dataset_handler import DatasetHandler
import pytorch_lightning as pl
from pathlib import Path
import sys
import os

class CoreSystem(pl.LightningModule):

    def __init__(self, param, param_trainer=None):
        super().__init__()

        self.p = param

        if param_trainer is not None:
            self.p_system = param_trainer
            self.param_trainer = param_trainer
            self.param = param
        else:
            self.p_system = param.system

        self.models, self.shapes, self.optimizers, self.schedulers = self.setup_module()
        self.d_handler = DatasetHandler(param)

    def validation_end(self, outputs):

        logs = {}

        for key in outputs[0].keys():
            logs[key] = torch.stack([x[key] for x in outputs]).mean()

        return {'log': logs, 'progress_bar': {'val_loss': logs['val_loss'].item()}}

    def configure_optimizers(self):
        return self.optimizers

    @pl.data_loader
    def train_dataloader(self):
        # TODO put somwhere else. Wait for pytorch-lightning to update it
        self.logger.write_models(self.models, self.shapes)
        return self.d_handler.dataloader_train

    @pl.data_loader
    def val_dataloader(self):
        return self.d_handler.dataloader_val

    @pl.data_loader
    def test_dataloader(self):
        return self.d_handler.dataloader_test

    def seed(self):
        seed = self.p.train.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def setup_module(self):
        models = []
        shapes = []
        optimizers = []
        schedulers = []

        for block_key in self.p_system:
            model_keys = [key for key in self.p_system[block_key] if 'model' in key]
            block_key = AttrDict(self.p_system[block_key])

            model_params = []

            for model_key in model_keys:
                model_key = block_key[model_key]
                repeat = model_key.model_params.repeat if 'repeat' in model_key.model_params else 1

                for _ in range(repeat):
                    model, shape = self.create_model(self.p, model_key.model_params)
                    model_params.extend(list(model.parameters()))
                    models.append(model)
                    shapes.append(shape)

            optimizer = self.get_optimizer(filter(lambda p: p.requires_grad, model_params), block_key.optimizer_params)
            scheduler = self.get_scheduler(optimizer, block_key.scheduler_params)
            schedulers.append(scheduler)
            optimizers.append(optimizer)

        return models, shapes, optimizers, schedulers

    def create_model(self, param, model_params):

        module_name = model_params.name
        model_type = model_params.type
        shape = model_params.shape_in

        args = [param, model_params]

        try:
            module = importlib.import_module(module_name)
            instance = getattr(module, model_type)
            model = instance(*args).to(self.p.device)
            shapes = [tuple(l) for l in shape]
            return model, shapes

        except Exception:
            raise Exception('{}:{} is not configured properly'.format(module_name, model_type))

    def get_optimizer(self, model_params, model_hyperparams):

        optimizer_type = model_hyperparams.name
        lr = model_hyperparams.lr

        if 'weight_decay' in model_hyperparams:
            weight_decay = model_hyperparams.weight_decay
        else:
            weight_decay = 0.0

        if optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adam':

            if 'betas' in model_hyperparams:
                betas = model_hyperparams.betas
            else:
                betas = (0.9, 0.999)

            optimizer = torch.optim.Adam(model_params, lr=lr, betas=betas, weight_decay=weight_decay)

        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model_params, lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError('{} is not implemented yet'.format(optimizer_type))

        return optimizer

    def get_scheduler(self, optimizer, scheduler_params):

        scheduler = None
        scheduler_type = scheduler_params.name

        if scheduler_type == 'step_lr':
            gamma = scheduler_params.gamma
            stepsize = scheduler_params.stepsize
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, stepsize, gamma, last_epoch=-1)
        if scheduler_type == 'one_cycle_lr':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.0001, epochs=1000, steps_per_epoch=5000, anneal_strategy='linear', last_epoch=-1)

        return scheduler

    def log_gradients(self):
        for model in self.models:
            self.logger.log_gradients(model)

    def get_loss_type(self, type):
        if type == 'mse':
            return torch.nn.MSELoss()
        elif type == 'l1':
            return torch.nn.L1Loss()
        else:
            raise NotImplementedError

    @classmethod
    def load_from_checkpoint(cls, model_save_path, param):

        try:
            weights = [x for x in Path(model_save_path).glob('*.ckpt')]
            file = max(weights, key=os.path.getctime)

            checkpoint = torch.load(file)

            model = cls(param)
            model.load_state_dict(checkpoint['state_dict'])

            # give model a chance to load something
            model.on_load_checkpoint(checkpoint)
            print('checkpoint loaded', file)
            return model
        except FileNotFoundError:
            print('No checkpoint file available: {}/checkpoint.pkl'.format(file))
            quit()
        except ValueError:
            print('No checkpoint found')
            quit()
