import torch
import utils.io as io
import utils.utils as helper
from utils.torchsummary import summary
import os
import yaml
from orderedattrdict.yamlutils import from_yaml
import torchvision
import utils.utils as utils
from pathlib import Path
import json
import numpy as np
from pytorch_lightning.logging import TestTubeLogger

yaml.add_constructor(u'tag:yaml.org,2002:map', from_yaml)
yaml.add_constructor(u'tag:yaml.org,2002:omap', from_yaml)

from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only

class Logger(TestTubeLogger):
    def __init__(self, param):
        super().__init__(save_dir=param.root_dir, name=param.experiment_name, version=param.version, debug=False)

        self.experiment.tag(utils.dict_to_keyvalue(param))
        self.hparam = utils.dict_to_keyvalue(param)
        self.param = param

        self.run_dir = Path(self.param.root_dir, self.param.experiment_name, 'version_{}'.format(self.param.version))
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_files_dir = self.run_dir / 'logs'
        self.log_files_dir.mkdir(exist_ok=True)
        self.results_files_dir = self.run_dir / 'results'

        self.results_files_dir.mkdir(exist_ok=True)
        self.log_tensorboard_dir = self.run_dir / 'tf'
        self.log_tensorboard_dir.mkdir(exist_ok=True)

        self.evaluation_list = []

        self.logging_files = False
        self.n_total_iterations = 0

        self.write_config()

    def write_models(self, models, shapes):
        # generate model file
        model_filename = open('{}/model.txt'.format(self.log_files_dir), "a")
        model_filename.seek(0)
        model_filename.truncate()

        for model, shape in zip(models, shapes):
            if model is not None:
                summary(model, input_size=shape, batch_size=2, device=self.param.device, file=model_filename)

        model_filename.close()

    def write_config(self):

        # generate config file
        config_filename = '{}/config.txt'.format(self.log_files_dir)
        config = open(config_filename, "a")
        config.seek(0)
        config.truncate()
        with open(config_filename, 'w') as yaml_file:
            # to_yaml(yaml, self.param)

            yaml.dump(self.param, yaml_file)
        config.flush()
        config.close()

    def evaluation_write(self, evaluation_keys, evaluation_values):

        evaluation_values = np.array(evaluation_values)

        evaluation_json = {}

        for idx, key in enumerate(evaluation_keys):
            evaluation_json[key] = list(evaluation_values[:, idx])

        filename = Path(self.results_files_dir / 'evaluation.json')

        # Writing JSON data
        with open(str(filename), 'w') as f:
            json.dump(evaluation_json, f)

    def log_file(self, global_step):

        if global_step % self.param.logger.log_files_every_n_iter == 0 and global_step > 0:
            return True
        else:
            return False

    def log_video(self, tag, frames, global_step, force=False):
        if self.log_file(global_step) or force:
            video_stack = torch.stack(frames, dim=1)
            video_stack = torch.clamp(video_stack, min=0, max=1)
            video_stack = (video_stack * 255).long()

            self.experiment.add_video('{}'.format(tag), video_stack, global_step)

    def log_image(self, tag, image, global_step, force=False):
        if self.log_file(global_step)or force:
            image = torch.clamp(image, min=0, max=1)
            self.experiment.add_images('{}'.format(tag), torch.clamp(image, min=0, max=1), global_step)

    def log_multi_channel_image(self, tag, multi_image, global_step, force=False):

        if self.log_file(global_step) or force:

            channels = multi_image.shape[1]

            for i in range(channels // 3):
                image = multi_image[:, i * 3:i * 3 + 3]
                image = torch.clamp(image, min=0, max=1)
                self.experiment.add_images('{}_{}'.format(tag, i), torch.clamp(image, min=0, max=1), global_step)

    def log_embedding(self, tag, labels, imgs, points, global_step, force=False):

        if self.log_file(global_step) or force:
            self.experiment.add_embedding(points, metadata=labels, label_img=imgs, global_step=global_step,
                                          tag='{}_{}'.format(self.param.name, tag))
