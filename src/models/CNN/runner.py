import sys
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from datasets import WaveformIndexDataModule
from models.CNN.CNN_lightning import CNNModule

class Runner:
    def __init__(
            self,
            project_name: str = 'fiducial-points',
            # data hyperparameters
            data_dir: str = 'data/',
            data_fname: str = 'wvfIndexData.pkl',
            batch_size: int = 32,
            size: dict = {'train': 0.8, 'val': 0.1, 'test': 0.1},
            seed: int = 3,
            split_type: str = 'subid',
            num_workers: int = 1,
            # model hyperparameters
            input_dim: int = 1000,
            output_dim: int = 2,
            layer_dims: list = [256, 128, 64],
            kernel_dim: int = 3,
            stride: int = 1,
            padding: int = 1,
            dropout: float = 0.35,
            activation_type: str = 'relu',
            lr: float = 1e-3,
            monitor_metric: str = 'val_loss',
            # trainer hyperparameters
            max_epochs: int = 100,
            gradient_clip_val: float = 10.0,
            gradient_clip_algorithm: str = 'value',
            deterministic: bool = True,
            devices: str = "auto",
            accelerator: str = "cpu",
            # other hyperparameters
            patience: int = 10,
        ):
        seed_everything(seed, workers=True)

        self.project_name = project_name

        self.data_hyperparameters = {
            'data_dir': data_dir,
            'data_fname': data_fname,
            'batch_size': batch_size,
            'size': size,
            'seed': seed,
            'split_type': split_type,
            'num_workers': num_workers,
        }

        self.model_hyperparameters = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'layer_dims': layer_dims,
            'kernel_dim': kernel_dim,
            'stride': stride,
            'padding': padding,
            'dropout': dropout,
            'activation_type': activation_type,
            'lr': lr,
            'monitor_metric': monitor_metric,
        }

        self.trainer_hyperparameters = {
            'max_epochs': max_epochs,
            'gradient_clip_val': gradient_clip_val,
            'gradient_clip_algorithm': gradient_clip_algorithm,
            'deterministic': deterministic,
            'devices': devices,
            'accelerator': accelerator,
        }

        self.other_hyperparameters = {
            'patience': patience,
        }

        self.run()

    def run(self):
        # initialize the logger
        wandb.init(project=self.project_name)
        wandb.config.update(self.data_hyperparameters)
        wandb.config.update(self.model_hyperparameters)
        wandb.config.update(self.trainer_hyperparameters)
        wandb.config.update(self.other_hyperparameters)
        wandb_logger = WandbLogger()

        # Set callbacks for trainer (lr monitor, early stopping)
        lr_monitor = LearningRateMonitor(logging_interval='step')       # Create a learning rate monitor callback
        early_stopping = EarlyStopping(
            monitor=self.model_hyperparameters['monitor_metric'], 
            patience=self.other_hyperparameters['patience'], 
            mode='min', 
            verbose=False
            )   # Create an early stopping callback
        checkpoint_callback = ModelCheckpoint(
            monitor=self.model_hyperparameters['monitor_metric'],
            mode='min',
            save_top_k=1,
            dirpath='checkpoints',
            filename='best-checkpoint',
        )
        callbacks = [lr_monitor, early_stopping, checkpoint_callback]    # aggregate all callbacks
        
        # initialize the trainer
        trainer = Trainer(
            logger=wandb_logger, # comment out to disable wandb logging
            callbacks=callbacks, 
            **self.trainer_hyperparameters
        )

        # load the data module
        data_module = WaveformIndexDataModule(**self.data_hyperparameters)
        data_module.setup()
        subid_split = data_module.return_subid_split()
        self.model_hyperparameters.update({'subid_split':subid_split})

        # initialize the model
        model = CNNModule(**self.model_hyperparameters)

        # train the model
        trainer.fit(model, datamodule=data_module)

        # test the model
        trainer.test(model, datamodule=data_module)#, ckpt_path='best')

        # terminate the wandb run
        wandb.finish()