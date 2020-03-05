from pathlib import Path
from pytorch_lightning import Trainer
from utils.io import load_config
import utils.utils as utils
from systems import SystemNeuralTexture
from test_tube import Experiment
from utils.logger import Logger
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import csv
import collections
import argparse

parser = argparse.ArgumentParser(description='Evaluates all trained models given the root path')
parser.add_argument('--trained_model_path', type=str, default='../trained_models', help='Path to root of trained models')

args = parser.parse_args()
root_dir = Path(args.trained_model_path)

experiment_name = 'neural_texture'

experiment_dir = root_dir / experiment_name
version_dirs = [x for x in experiment_dir.iterdir() if x.is_dir()]

for idx, version_dir in enumerate(version_dirs):
    config_path = version_dir / 'logs' / 'config.txt'
    param = load_config(root_dir, config_path)
    param['train']['bs'] = 1
    param['experiment_name'] = experiment_name
    system = SystemNeuralTexture(param)
    logger = Logger(param)

    exp = Experiment(name=param.experiment_name, save_dir=param.root_dir, version=param.version, debug=False)
    exp.tag(utils.dict_to_keyvalue(param))

    trainer = Trainer(logger=logger, gpus=1, test_percent_check=1.0)

    system = system.load_from_checkpoint(str(logger.run_dir / 'checkpoints'), param)
    trainer.test(system)