import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb

from models.MLP.MLP_pytorch import MLP

class MLPModule(pl.LightningModule):
    def __init__(self,
                 input_dim: int = 1000,
                 output_dim: int = 3,
                 layer_dims: list = [128, 64, 32],
                 dropout: float = 0.35,
                 activation_type: str = 'relu',
                 lr: float = 1e-3,
                 monitor_metric: str = 'val_loss',
                 seed: int = 3,
                 **kwargs
                 ):
        super(MLPModule, self).__init__()

        # create model instance
        self.model = MLP(input_dim=input_dim,
                         output_dim=output_dim,
                         layer_dims=layer_dims,
                         dropout=dropout,
                         activation_type=activation_type
                         )
        
        # fix the random seed for repeatability
        torch.manual_seed(seed)

        self.learning_rate = lr
        self.monitor_metric = monitor_metric
        self.losses = {'train': [], 'val': [], 'test': []}
        self.kwargs = kwargs

    def forward(self, x):
        return self.model(x)
    
    def custom_loss(self, y_hat, y):
        loss = F.mse_loss(y_hat, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y, index, subid, lengths, wvfs = batch
        y_hat = self.forward(x)
        loss = self.custom_loss(y_hat, y)
        self.losses['train'].append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, index, subid, lengths, wvfs = batch
        y_hat = self.forward(x)
        loss = self.custom_loss(y_hat, y)
        self.losses['val'].append(loss.item())
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, index, subid, lengths, wvfs = batch
        y_hat = self.forward(x)
        loss = self.custom_loss(y_hat, y)
        self.losses['test'].append(loss.item())
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def on_train_epoch_end(self):
        # log the average loss for the training set
        train_losses = np.mean(self.losses['train'])
        self.log('loss/train/mse', torch.mean(torch.tensor(train_losses)))
        self.losses['train'].clear()

    def on_validation_epoch_end(self):
        # skip logging if we are in the sanity checking
        if self.trainer.sanity_checking:
            return

        # log the average loss for the validation set
        val_losses = np.mean(self.losses['val'])
        self.log('loss/val/mse', torch.mean(torch.tensor(val_losses)))
        self.losses['val'].clear()

    def on_test_epoch_end(self):

        # log the average loss for the test set
        test_losses = np.mean(self.losses['test'])
        self.log('loss/test/mse', torch.mean(torch.tensor(test_losses)))
        self.losses['test'].clear()

        all_x, all_y, all_y_hat, paramlist = self.calculate_metrics(self.trainer.test_dataloaders)
        self.make_waveform_index_plots(all_x, all_y, all_y_hat, 'test')
        self.make_prediction_accuracy_plots(paramlist, 'test')

    def on_train_end(self):
        print('On train end ...')
        
        all_x, all_y, all_y_hat, paramlist = self.calculate_metrics(self.trainer.train_dataloader)
        self.make_waveform_index_plots(all_x, all_y, all_y_hat, 'train')
        self.make_prediction_accuracy_plots(paramlist, 'train')

        all_x, all_y, all_y_hat, paramlist = self.calculate_metrics(self.trainer.val_dataloaders)
        self.make_waveform_index_plots(all_x, all_y, all_y_hat, 'val')
        self.make_prediction_accuracy_plots(paramlist, 'val')
    
    def calculate_metrics(self, dataloader):
        all_x, all_y, all_y_hat, all_lengths, all_wvfs = [],[],[],[],[]
        for batch in dataloader:
            x, y, index, subid, lengths, wvfs = batch
            x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)
            
            # get the predictions
            y_hat = self.model(x)

            # append the predictions to the list
            all_x.extend(x)
            all_y.extend(y)
            all_y_hat.extend(y_hat)
            all_lengths.extend(lengths)
            all_wvfs.extend(wvfs)

        # convert the lists to tensors
        all_x = torch.stack(all_x)
        all_y = torch.stack(all_y)
        all_y_hat = torch.stack(all_y_hat)
        all_lengths = torch.stack(all_lengths)
        all_wvfs = torch.stack(all_wvfs)
        
        # calculate mse
        mse = F.mse_loss(all_y, all_y_hat)

        # calculate physio params
        paramlist = self.calculate_physio_params(all_wvfs, all_y, all_y_hat, all_lengths)
        
        return all_x, all_y, all_y_hat, paramlist
        
    def calculate_physio_params(self, all_x, all_y, all_y_hat, all_lengths):

        # unpack p1 and p2 from the tensor
        all_x = all_x.cpu().detach().numpy()
        all_y = all_y.cpu().detach().numpy()
        all_y_hat = all_y_hat.cpu().detach().numpy()
        all_lengths = all_lengths.cpu().detach().numpy()
        true_p1, true_p2, true_n = all_y[:, 0], all_y[:, 1], all_y[:, 2]
        pred_p1, pred_p2, pred_n = all_y_hat[:, 0], all_y_hat[:, 1], all_y_hat[:, 2]

        # convert the percentage units to index units
        true_p1 = (true_p1 * all_lengths).astype(int)
        true_p2 = (true_p2 * all_lengths).astype(int)
        true_n = (true_n * all_lengths).astype(int)
        pred_p1 = (pred_p1 * all_lengths).astype(int)
        pred_p2 = (pred_p2 * all_lengths).astype(int)
        pred_n = (pred_n * all_lengths).astype(int)

        # calculate the augmentation index (AIX)
        all_aix_true, all_aix_pred = [],[]
        for x, p1t, p2t, p1p, p2p in zip(all_x, true_p1, true_p2, pred_p1, pred_p2):
            x = x[~np.isnan(x)]
            aix_true = 100*(x[p2t] - x[p1t]) / x.ptp()
            aix_pred = 100*(x[p2p] - x[p1p]) / x.ptp()
            all_aix_true.append(aix_true)
            all_aix_pred.append(aix_pred)

        # calculate the spti parameter
        all_spti_true, all_spti_pred = [],[]
        for x, n_ind_t, n_ind_p in zip(all_x, true_n, pred_n):
            spti_true = x[:n_ind_t].sum() / 1000
            spti_pred = x[:n_ind_p].sum() / 1000
            all_spti_true.append(spti_true)
            all_spti_pred.append(spti_pred)

        # calculate the end systolic pressure (ESP)
        all_esp_true, all_esp_pred = [],[]
        for x, n_ind_t, n_ind_p in zip(all_x, true_n, pred_n):
            esp_true = x[n_ind_t]
            esp_pred = x[n_ind_p]
            all_esp_true.append(esp_true)
            all_esp_pred.append(esp_pred)

        # combine a parameter list
        paramlist = [
            [np.array(true_p1), np.array(pred_p1), 'p1_ind'],
            [np.array(true_p2), np.array(pred_p2), 'p2_ind'],   
            [np.array(true_n), np.array(pred_n), 'n_ind'],
            [np.array(all_aix_true), np.array(all_aix_pred), 'AIX'],
            [np.array(all_spti_true), np.array(all_spti_pred), 'SPTI'],
            [np.array(all_esp_true), np.array(all_esp_pred), 'ESP'],
        ]

        return paramlist

    def make_waveform_index_plots(self, all_x, all_y, all_y_hat, stage):
        
        # get 10 random examples
        indices = np.random.choice(range(all_x.size(0)), 10)
        xs = all_x[indices].cpu().detach().numpy()
        ys = all_y[indices].cpu().detach().numpy()
        ys_hat = all_y_hat[indices].cpu().detach().numpy()

        # plot the examples
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for ax, xs, ys, y_hat in zip(axes.ravel(), xs, ys, ys_hat):
            ax.plot(np.linspace(0, 1, len(xs.squeeze())), xs.squeeze())
            ax.axvline(ys[0], color='r', linestyle='-', alpha=0.5, label='P1')
            ax.axvline(ys[1], color='k', linestyle='-', alpha=0.5, label='P2')
            ax.axvline(ys[2], color='g', linestyle='-', alpha=0.5, label='N')
            ax.axvline(y_hat[0], color='r', linestyle='--')
            ax.axvline(y_hat[1], color='k', linestyle='--')
            ax.axvline(y_hat[2], color='g', linestyle='--')
        
        plt.tight_layout()
        wandb.log({f"params/{stage}/index_wvf_plot": wandb.Image(fig)})
        plt.close(fig)
        
    def make_prediction_accuracy_plots(self, paramlist, stage):
        
        num_params = len(paramlist)
        # log metrics for the parameter pairs
        fig = plt.figure(figsize=(3*num_params, 8))
        gs = fig.add_gridspec(1, num_params)
        
        for i in range(num_params):
            true, pred, name = paramlist[i]
            
            # create the true-vs-pred and bland-altman plots
            gs0 = gs[i].subgridspec(2, 1)
            ax0 = fig.add_subplot(gs0[0])
            ax1 = fig.add_subplot(gs0[1])
            self.make_physio_param_sub_figs(pred, true, ax0, ax1, name, decs=0)

        plt.tight_layout()
        wandb.log({f"params/{stage}/physio_param_plot": wandb.Image(fig)})
        plt.close(fig)

    def make_physio_param_sub_figs(self, pred, true, ax0, ax1, name, decs=1):
        """ Creates two subplot with the true vs predicted values and the bland-altman plot. """

        # create the true vs pred plot
        ax0.scatter(pred, true, marker='o', facecolor='none', edgecolor='tab:blue')
        ax0.set_title(f'{name}', fontsize=14)
        ax0.set_xlabel('Predicted')
        ax0.set_ylabel('True')
        ax0.spines[['right','top']].set_visible(False)
        lims = [
            np.min([ax0.get_xlim()[0], ax0.get_ylim()[0]]) * 0.95,
            np.max([ax0.get_xlim()[1], ax0.get_ylim()[1]]) * 1.05,
        ]
        ax0.plot(lims, lims, linestyle='-',color='black', alpha=0.75, zorder=0, linewidth=1.25)
        ax0.set_xlim(lims)
        ax0.set_ylim(lims)

        # calculate r correlation coefficient and write it on the olot
        text = f'n={len(pred)}\nr={np.corrcoef(pred, true)[0, 1]:.2}'
        ax0.text(0.95, 0.05, text, transform=ax0.transAxes, fontsize=10, ha='right', va='bottom',)

        # create the bland-altman plot
        mean = 0.5*(true + pred)
        diff = true - pred
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        u_loa = mean_diff + 1.96*std_diff
        l_loa = mean_diff - 1.96*std_diff
        ax1.scatter(mean, diff, marker='o', facecolor='none', edgecolor='tab:blue')
        ax1.axhline(mean_diff, color='tab:red', linestyle='-', linewidth=1.5, zorder=10)
        ax1.axhline(u_loa, color='tab:red', linestyle='--', linewidth=1.5, zorder=10, alpha=0.6)
        ax1.axhline(l_loa, color='tab:red', linestyle='--', linewidth=1.5, zorder=10, alpha=0.6)
        ax1.fill_between(ax1.get_xlim(), u_loa, l_loa, color='tab:red', alpha=0.1, edgecolor='none')
        ax1.set_xlabel('Average')
        ax1.set_ylabel('Difference')
        ax1.spines[['right','top']].set_visible(False)

        # write the mean difference and the 95% limits of agreement
        min_ = np.min([np.min(true), np.min(pred)])
        min_ -= 0.1*np.abs(min_)
        if decs == 0:
            txt_1 = f'{int(np.round(mean_diff, decs)):.2f}\n'
            txt_2 = f'{int(np.round(mean_diff+1.96*std_diff, decs))}\n'
            txt_3 = f'{int(np.round(mean_diff-1.96*std_diff, decs))}\n'
        else:
            txt_1 = f'{float(np.round(mean_diff, decs)):.2f}\n'
            txt_2 = f'{float(np.round(mean_diff+1.96*std_diff, decs))}\n'
            txt_3 = f'{float(np.round(mean_diff-1.96*std_diff, decs))}\n'
        x_pos = ax1.get_xlim()[1] - 0.02*(np.diff(ax1.get_xlim()))
        ax1.text(x_pos, mean_diff, txt_1, fontsize=9, color='tab:red', ha='right', va='center')
        ax1.text(x_pos, mean_diff+1.96*std_diff, txt_2, fontsize=9, color='tab:red', ha='right', va='center')
        ax1.text(x_pos, mean_diff-1.96*std_diff, txt_3, fontsize=9, color='tab:red', ha='right', va='center')
        
        # resize plot to fit everything
        ax1.set_ylim(
          np.min([ax1.get_ylim()[0], 1.3*l_loa]),
          np.max([ax1.get_ylim()[1], 1.3*u_loa]),
          )