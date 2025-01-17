import os

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb

from models.CNN.CNN_pytorch import CNN

class CNNLightning(pl.LightningModule):
    def __init__(self, hparams: dict):
        super(CNNLightning, self).__init__()
        self.hparams = hparams
        self.model = CNN(input_dim=hparams['input_dim'],
                         output_dim=hparams['output_dim'],
                         layer_dims=hparams['layer_dims'],
                         kernel_dim=hparams['kernel_dim'],
                         stride=hparams['stride'],
                         padding=hparams['padding'],
                         dropout=hparams['dropout'],
                         activation_type=hparams['activation_type']
                         )
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])
        return optimizer