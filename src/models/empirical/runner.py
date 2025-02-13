import sys
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer, seed_everything

from datasets import WaveformIndexDataModule
from models.empirical.empirical_method import empirical_method

class Runner:
    def __init__(
            self,
            project_name: str = 'fiducial-points',
            model_name: str = 'empirical',
            # data hyperparameters
            data_dir: str = 'data/',
            data_fname: str = 'wvfIndexData_AO.pkl',
            batch_size: int = 32,
            size: dict = {'train': 0.8, 'val': 0.1, 'test': 0.1},
            seed: int = 3,
            split_type: str = 'subid',
            num_workers: int = 1,
            train_proportion: float = 1.0,
            train_proportion_seed: int = None,
            # model hyperparameters
            special_test: str = None,
            # trainer hyperparameters
            # other hyperparameters
        ):
        seed_everything(seed, workers=True)

        self.project_name = project_name
        self.model_name = model_name

        self.data_hyperparameters = {
            'data_dir': data_dir,
            'data_fname': data_fname,
            'batch_size': batch_size,
            'size': size,
            'seed': seed,
            'split_type': split_type,
            'num_workers': num_workers,
            'train_proportion': train_proportion,
            'train_proportion_seed': train_proportion_seed,
        }

        self.model_hyperparameters = {
        }

        self.trainer_hyperparameters = {
        }

        self.other_hyperparameters = {
        }

        self.run()

    def run(self):
        from matplotlib import pyplot as plt

        # load the data module
        data_module = WaveformIndexDataModule(**self.data_hyperparameters)
        data_module.setup()
        subid_split = data_module.return_subid_split()
        
        # run through the testing population for the empirical model
        results = []
        for arr in data_module.test_dataloader():
            inputs, outputs, indices, subids, lengths, wvfs = arr
            
            # convert tensors to numpy
            lengths = lengths.numpy()
            wvfs = wvfs.numpy()
            outputs = outputs.numpy()
            indices = indices.numpy()
            for wvf, idx, output, len_ in zip(wvfs, indices, outputs, lengths):
                # unpack the iteration data
                wvf = wvf[~np.isnan(wvf)]   # remove nans from the waveform
                subid, loc, cycle = idx
                subid = str(subid)

                # format subid
                subid = '0'*(6-len(subid)) + subid
                SiteID = subid[3:]
                SUBID = subid[:3]
                
                # calculate the indices
                p_ind, i_ind, n_ind = empirical_method(wvf)
            
                # convert outputs from tensor to numpy
                output *= len_
                output = output.round(0).astype(int)

                # if i_ind is not None and np.abs(output[0] - i_ind) > 250:
                #     print(output[0], i_ind)
                #     plt.plot(wvf)
                #     plt.axvline(x=output[0], color='r')
                #     plt.axvline(x=i_ind, color='g')
                #     plt.show()

                # store results
                # print(f'Index: {idx}, Output: {output}, P1: {p1_ind}, P2: {p2_ind}, N: {n_ind}')

                results.append([SUBID, SiteID, loc, idx[2], output[0], output[1], i_ind, n_ind, wvf])
            
        # save the results
        results = pd.DataFrame(results, columns=['SUBID', 'SiteID', 'loc', 'cycle', 'i_true', 'n_true', 'i_pred', 'n_pred','waveform'])
        results.set_index(['SUBID','SiteID','loc','cycle'], inplace=True)
        
        # save the results
        results.to_pickle('./save/results_base.pkl')
        