import os

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset
from scipy.signal import resample

def data_preprocessing(df):
    """ Function preprocesses the data for input to the CNN model. 
    The preprocessing involves normalizing of the 
        1. input waveforms in time and amplitude.

        2. output labels to be in the range [0, 1]
    """

    # normalize the waveforms lengths to 1000
    df['wvf_norm'] = df.apply(lambda x: resample(x['wvf'], 1000), axis=1)
    
    # normalize the waveforms to have zero mean and unit variance
    df['wvf_norm'] = df['wvf_norm'].apply(lambda x: (x - np.mean(x)) / np.std(x))

    # normalize the labels to be in the range [0, 1]
    df['p1_ind_norm'] = df['p1_ind'].div(df['m_ind'])
    df['p2_ind_norm'] = df['p2_ind'].div(df['m_ind'])
    df['n_ind_norm'] = df['n_ind'].div(df['m_ind'])

    # convert the SUBID, SiteID, etc to a single string
    df['index'] = df.apply(lambda x: (int(x.name[0]+x.name[1]), x['loc_id'], x.name[3]), axis=1)
    df['subid'] = df.apply(lambda x: int(x.name[0]+x.name[1]), axis=1)

    # # Debugging - plot an example waveform
    # print('mean:', np.mean(df['wvf_norm'].iloc[0]), 'std:', np.std(df['wvf_norm'].iloc[0]))
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(1, 2, figsize=(12,6))
    # ax[0].plot(df['wvf'].iloc[0])
    # ax[0].axvline(df['p1_ind'].iloc[0], color='r', linestyle='--')
    # ax[0].axvline(df['p2_ind'].iloc[0], color='k', linestyle='--')
    # ax[0].set_title('Base Waveform')
    # ax[1].plot(np.linspace(0, 1, 1000), df['wvf_norm'].iloc[0])
    # ax[1].axvline(df['p1_ind_norm'].iloc[0], color='r', linestyle='--')
    # ax[1].axvline(df['p2_ind_norm'].iloc[0], color='k', linestyle='--')
    # ax[1].set_title('Normalized Waveform')
    # plt.tight_layout()
    # plt.show()

    # convert structures to Tensor
    inputs = torch.tensor([list(x) for x in df['wvf_norm'].values], dtype=torch.float32)
    outputs = torch.tensor(df[['p1_ind_norm', 'p2_ind_norm']].values, dtype=torch.float32)
    indices = torch.tensor([df['index'].values.tolist()], dtype=torch.int64).squeeze()
    subids = torch.tensor(df['subid'].values, dtype=torch.int64)
    print(inputs.shape, outputs.shape, indices.shape, subids.shape)

    return inputs, outputs, indices, subids

class StringDataset(Dataset):
    def __init__(self, array_of_strings):
        self.data = array_of_strings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class WaveformIndexDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = 'data/', 
                 data_fname: str = 'waveform_data.pkl',
                 batch_size: int = 32,
                 size: dict = {'train': 0.8, 'val': 0.1, 'test': 0.1},
                 num_workers: int = 1,
                 seed: int = 3,
                 split_type: str = 'subid',
                ):
                 
        super().__init__()
        self.data_dir = data_dir
        self.data_fname = data_fname
        self.batch_size = batch_size
        self.size = size
        self.num_workers = num_workers
        self.seed = seed
        self.generator = torch.Generator().manual_seed(self.seed)
        self.split_type = split_type
        self.subid_split = {'train': None, 'val': None, 'test': None}
        
    def prepare_data(self):
        # load the dataset
        df = pd.read_pickle(os.path.join(self.data_dir, self.data_fname))
        
        # preprocess the dataset to normalize the waveforms for input to the CNN
        inputs, outputs, indices, subids = data_preprocessing(df)
        
        return inputs, outputs, indices, subids

    def setup(self, stage=None):
        # Assign train, validation, and test datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            inputs, outputs, indices, subids = self.prepare_data()
            
            if self.split_type == 'cardiac_cycle':
                dataset = TensorDataset(inputs, outputs, indices, subids)
                self.train_data, self.val_data, self.test_data = random_split(dataset, [self.size['train'], self.size['val'], self.size['test']], generator=self.generator)
            elif self.split_type == 'subid':
                # split the datasets according to subids to avoid data leakage
                unique_subids = StringDataset(subids.unique())
                train_, val_, test_ = random_split(unique_subids, [self.size['train'], self.size['val'], self.size['test']], generator=self.generator)
                train_subids = unique_subids[train_.indices]
                val_subids = unique_subids[val_.indices]
                test_subids = unique_subids[test_.indices]
                
                # store the subid split for later use
                self.subid_split['train'] = train_subids
                self.subid_split['val'] = val_subids
                self.subid_split['test'] = test_subids

                # split the data into training and validation sets using subid values
                inds_train = [i for i, subid in enumerate(subids) if subid in train_subids]
                inds_val = [i for i, subid in enumerate(subids) if subid in val_subids]
                inds_test = [i for i, subid in enumerate(subids) if subid in test_subids]
                
                inputs_train = inputs[inds_train]
                outputs_train = outputs[inds_train]
                indices_train = indices[inds_train]
                subids_train = subids[inds_train]
                self.train_data = TensorDataset(inputs_train, outputs_train, indices_train, subids_train)
                
                inputs_val = inputs[inds_val]
                outputs_val = outputs[inds_val]
                indices_val = indices[inds_val]
                subids_val = subids[inds_val]
                self.val_data = TensorDataset(inputs_val, outputs_val, indices_val, subids_val)

                inputs_test = inputs[inds_test]
                outputs_test = outputs[inds_test]
                indices_test = indices[inds_test]
                subids_test = subids[inds_test]
                self.test_data = TensorDataset(inputs_test, outputs_test, indices_test, subids_test)
            else:
                print("unknown split type")

            print('Train:', len(self.train_data), 'Val:', len(self.val_data), 'Test:', len(self.test_data))


    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size=self.batch_size,
                          shuffle=True,
                          generator=self.generator,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0
                          )

    def val_dataloader(self):
        return DataLoader(self.val_data, 
                          batch_size=self.batch_size,
                          shuffle=False,
                          generator=self.generator,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0
                          )

    def test_dataloader(self):
        return DataLoader(self.test_data, 
                          batch_size=self.batch_size,
                          shuffle=False,
                          generator=self.generator,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0
                          )