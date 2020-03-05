from pytorch_lightning import Trainer
from utils.io import load_config_train
import utils.utils as utils
from systems import SystemNeuralTexture
from test_tube import Experiment
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from utils.logger import Logger

if __name__ == '__main__':
    experiment_name = 'neural_texture'
    root_dir = Path().cwd() / '..' / 'trained_models'
    param = load_config_train(root_dir, experiment_name)
    logger = Logger(param)
    system = SystemNeuralTexture(param)

    checkpoint = ModelCheckpoint(
        filepath=str(logger.run_dir / 'checkpoints'),
        save_top_k=5,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=param.experiment_name
    )

    trainer = Trainer(
        logger=logger,
        weights_summary=None,
        max_epochs=param.train.epochs,
        checkpoint_callback=checkpoint,
        early_stop_callback=None,
        row_log_interval=param.logger.log_scalars_every_n_iter,
        check_val_every_n_epoch=param.logger.log_validation_every_n_epochs,
        val_check_interval=1.0,
        accumulate_grad_batches=param.train.accumulate_grad_batches,
        train_percent_check=1.0,
        val_percent_check=1.0,
        test_percent_check=1.0,
        track_grad_norm=2,
        gpus=param.n_gpus,
        fast_dev_run=False
    )
    trainer.fit(system)
