import os

import torch
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset
from scipy.signal import resample

from utils.butterworth_filter import butterworth_filter

def data_preprocessing(df):
    """ Function preprocesses the data for input to the CNN model. 
    The preprocessing involves normalizing of the 
        1. input waveforms in time and amplitude.

        2. output labels to be in the range [0, 1]
    """
    # low pass filter the waveforms at 30 Hz
    df['wvf'] = df['wvf'].apply(lambda x: butterworth_filter(x, 30, 1000, 4, type_='low'))

    # normalize the waveforms
    df['wvf_norm'] = df.apply(lambda x: resample(x['wvf'], 1000), axis=1)               # normalize length to 1000
    df['wvf_norm'] = df['wvf_norm'].apply(lambda x: butterworth_filter(x, 40, 1000, 4, type_='low')) # low pass filter the waveforms at 30 Hz
    df['wvf_norm'] = df['wvf_norm'].apply(lambda x: (x - np.mean(x)) / np.std(x))       # normalize amplitude to zero mean and unit variance

    # calibrate waveforms to have amplitude in the [DBP, SBP] range
    df['wvf_calib'] = df.apply(lambda x: (x['sbp']-x['dbp']) * (x['wvf']-x['wvf'].min()) / (x['wvf'].ptp()) + x['dbp'], axis=1)

    # normalize the labels to be in the range [0, 1]
    # df['p1_ind_norm'] = df['p1_ind'].div(df['m_ind'])
    # df['p2_ind_norm'] = df['p2_ind'].div(df['m_ind'])
    df['p_ind_norm'] = df['p_ind'].div(df['m_ind'])
    df['i_ind_norm'] = df['i_ind'].div(df['m_ind'])
    df['n_ind_norm'] = df['n_ind'].div(df['m_ind'])

    # convert the SUBID, SiteID, etc to a single string
    df['index'] = df.apply(lambda x: (int(x.name[0]+x.name[1]), x['loc_id'], x.name[3]), axis=1)
    df['subid'] = df.apply(lambda x: int(x.name[0]+x.name[1]), axis=1)

    # # Debugging - plot an example waveform
    # print('mean:', np.mean(df['wvf_norm'].iloc[0]), 'std:', np.std(df['wvf_norm'].iloc[0]))
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(1, 3, figsize=(12,6))
    # ax[0].plot(df['wvf'].iloc[0])
    # ax[0].axvline(df['p1_ind'].iloc[0], color='r', linestyle='--')
    # ax[0].axvline(df['p2_ind'].iloc[0], color='k', linestyle='--')
    # ax[0].axvline(df['n_ind'].iloc[0], color='k', linestyle='--')
    # ax[0].set_title('Base Waveform')
    # ax[1].plot(np.linspace(0, 1, 1000), df['wvf_norm'].iloc[0])
    # ax[1].axvline(df['p1_ind_norm'].iloc[0], color='r', linestyle='--')
    # ax[1].axvline(df['p2_ind_norm'].iloc[0], color='k', linestyle='--')
    # ax[1].axvline(df['n_ind_norm'].iloc[0], color='k', linestyle='--')
    # ax[1].set_title('Normalized Waveform')
    # ax[2].plot(df['wvf_calib'].iloc[0])
    # ax[2].axvline(df['p1_ind'].iloc[0], color='r', linestyle='--')
    # ax[2].axvline(df['p2_ind'].iloc[0], color='k', linestyle='--')
    # ax[2].axvline(df['n_ind'].iloc[0], color='k', linestyle='--')
    # ax[2].set_title('Calibrated Waveform')
    # plt.tight_layout()
    # plt.show()

    # convert structures to Tensor
    inputs = torch.tensor([list(x) for x in df['wvf_norm'].values], dtype=torch.float32)
    outputs = torch.tensor(df[['i_ind_norm', 'n_ind_norm']].values, dtype=torch.float32)
    indices = torch.tensor([df['index'].values.tolist()], dtype=torch.int64).squeeze()
    subids = torch.tensor(df['subid'].values, dtype=torch.int64)
    lengths = torch.tensor(df['m_ind'].values, dtype=torch.int64)
    
    # create a padded tensor for the calibrated waveforms
    calib_wvf = torch.nn.utils.rnn.pad_sequence([torch.tensor(x, dtype=torch.float32) for x in df['wvf_calib'].values], padding_value=torch.nan, batch_first=True)
    print(inputs.size(), outputs.size(), indices.size(), subids.size(), lengths.size(), calib_wvf.size())

    return inputs, outputs, indices, subids, lengths, calib_wvf

def data_preprocessing_with_derivatives(df):
    """ Function preprocesses the data for input to the CNN model with multichannel inputs.
    The multichannel inputs will be the original waveform, the first derivative, and the second derivative.
    The preprocessing involves normalizing of the 
        1. input waveforms in time and amplitude.
        2. output labels to be in the range [0, 1]
    """

    # low pass filter the waveforms at 30 Hz
    df['wvf'] = df['wvf'].apply(lambda x: butterworth_filter(x, 30, 1000, 4, type_='low'))

    # # remove waveform trend
    # df['wvf'] = df['wvf'].apply(lambda x: x - np.linspace(0, x[-1] - x[0], len(x)))

    # normalize the waveforms
    df['wvf_norm'] = df.apply(lambda x: resample(x['wvf'], 1000), axis=1)               # normalize length to 1000
    df['wvf_norm'] = df['wvf_norm'].apply(lambda x: butterworth_filter(x, 40, 1000, 4, type_='low')) # low pass filter the waveforms at 30 Hz
    df['wvf_norm'] = df['wvf_norm'].apply(lambda x: (x - np.mean(x)) / np.std(x))       # normalize amplitude to zero mean and unit variance

    # compute the first and second derivatives of the normalized waveforms
    df['wvf_d_norm'] = df['wvf_norm'].apply(lambda x: np.gradient(x))
    df['wvf_d_norm'] = df['wvf_d_norm'].apply(lambda x: (x - np.mean(x)) / np.std(x))       # normalize amplitude to zero mean and unit variance
    df['wvf_dd_norm'] = df['wvf_d_norm'].apply(lambda x: np.gradient(x))
    df['wvf_dd_norm'] = df['wvf_dd_norm'].apply(lambda x: (x - np.mean(x)) / np.std(x))     # normalize amplitude to zero mean and unit variance
    df['wvf_ddd_norm'] = df['wvf_dd_norm'].apply(lambda x: np.gradient(x))
    df['wvf_ddd_norm'] = df['wvf_ddd_norm'].apply(lambda x: (x - np.mean(x)) / np.std(x))     # normalize amplitude to zero mean and unit variance
    df['wvf_dddd_norm'] = df['wvf_ddd_norm'].apply(lambda x: np.gradient(x))
    df['wvf_dddd_norm'] = df['wvf_dddd_norm'].apply(lambda x: (x - np.mean(x)) / np.std(x))     # normalize amplitude to zero mean and unit variance

    # calibrate waveforms to have amplitude in the [DBP, SBP] range
    df['wvf_calib'] = df.apply(lambda x: (x['sbp']-x['dbp']) * (x['wvf']-x['wvf'].min()) / (x['wvf'].ptp()) + x['dbp'], axis=1)

    # normalize the labels to be in the range [0, 1]
    # df['p1_ind_norm'] = df['p1_ind'].div(df['m_ind'])
    # df['p2_ind_norm'] = df['p2_ind'].div(df['m_ind'])
    df['p_ind_norm'] = df['p_ind'].div(df['m_ind'])
    df['i_ind_norm'] = df['i_ind'].div(df['m_ind'])
    df['n_ind_norm'] = df['n_ind'].div(df['m_ind'])

    # convert the SUBID, SiteID, etc to a single string
    df['index'] = df.apply(lambda x: (int(x.name[0]+x.name[1]), x['loc_id'], x.name[3]), axis=1)
    df['subid'] = df.apply(lambda x: int(x.name[0]+x.name[1]), axis=1)

    # # Debugging - plot an example waveform
    # print('mean:', np.mean(df['wvf_norm'].iloc[0]), 'std:', np.std(df['wvf_norm'].iloc[0]))
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(1, 3, figsize=(12,6))
    # ax[0].plot(df['wvf'].iloc[0])
    # ax[0].axvline(df['p1_ind'].iloc[0], color='r', linestyle='--')
    # ax[0].axvline(df['p2_ind'].iloc[0], color='k', linestyle='--')
    # ax[0].axvline(df['n_ind'].iloc[0], color='k', linestyle='--')
    # ax[0].set_title('Base Waveform')
    # ax[1].plot(np.linspace(0, 1, 1000), df['wvf_norm'].iloc[0])
    # ax[1].axvline(df['p1_ind_norm'].iloc[0], color='r', linestyle='--')
    # ax[1].axvline(df['p2_ind_norm'].iloc[0], color='k', linestyle='--')
    # ax[1].axvline(df['n_ind_norm'].iloc[0], color='k', linestyle='--')
    # ax[1].set_title('Normalized Waveform')
    # ax[2].plot(df['wvf_calib'].iloc[0])
    # ax[2].axvline(df['p1_ind'].iloc[0], color='r', linestyle='--')
    # ax[2].axvline(df['p2_ind'].iloc[0], color='k', linestyle='--')
    # ax[2].axvline(df['n_ind'].iloc[0], color='k', linestyle='--')
    # ax[2].set_title('Calibrated Waveform')
    # plt.tight_layout()
    # plt.show()

    # convert structures to Tensor
    inputs = torch.tensor([[list(x), list(dx), list(dxx)] for x, dx, dxx in df[['wvf_norm', 'wvf_d_norm', 'wvf_dd_norm']].values], dtype=torch.float32)
    outputs = torch.tensor(df[['i_ind_norm', 'n_ind_norm']].values, dtype=torch.float32)
    indices = torch.tensor([df['index'].values.tolist()], dtype=torch.int64).squeeze()
    subids = torch.tensor(df['subid'].values, dtype=torch.int64)
    lengths = torch.tensor(df['m_ind'].values, dtype=torch.int64)
    
    # create a padded tensor for the calibrated waveforms
    calib_wvf = torch.nn.utils.rnn.pad_sequence([torch.tensor(x, dtype=torch.float32) for x in df['wvf_calib'].values], padding_value=torch.nan, batch_first=True)
    print(inputs.size(), outputs.size(), indices.size(), subids.size(), lengths.size(), calib_wvf.size())

    return inputs, outputs, indices, subids, lengths, calib_wvf

def dataset_augmentation(dataset, aug_type, aug_prob):
    """ function is used to agument a dataset that will be used for the model training. """
    
    # fix the random seed for generating random integers
    random.seed(3)

    # unpack the dataset
    inputs, outputs, indices, subids, lengths, wvfs = dataset.tensors
    print(inputs.size(), outputs.size(), indices.size(), subids.size(), lengths.size(), wvfs.size())

    # determine number of channels in inputs
    num_channels = inputs.size(1)
    print('Number of channels : ', num_channels)

    inputs_aug, outputs_aug, indices_aug, subids_aug, lengths_aug, wvfs_avg = [], [], [], [], [], []
    for i, (input, output, index, subid, length, wvf) in enumerate(zip(inputs, outputs, indices, subids, lengths, wvfs)):
        
        # detach the tensors
        input = input.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        index = index.cpu().detach().numpy()
        subid = subid.cpu().detach().numpy()
        length = length.cpu().detach().numpy()
        wvf = wvf.cpu().detach().numpy()

        # augment data method 0: no augmentation
        if input.shape[0] == 1:
            inputs_aug.append([list(input[0])])
        elif input.shape[0] == 3:
            inputs_aug.append([list(x) for x in input])
        else:
            inputs_aug.append(input)
        outputs_aug.append(output)
        indices_aug.append(index)
        subids_aug.append(int(subid))
        lengths_aug.append(int(length))
        wvfs_avg.append(wvf)

        # augment data method 1: truncate the waveforms and resample to 1000
        if aug_type[0]:
            if random.randint(1, aug_prob[0]) == 1:
                trnc_len = random.randint(30, 150)
                output_aug1 = output * 1000 / (1000 - trnc_len)
                index_aug1 = index.copy()
                index_aug1[1] = 2
                subid_aug1 = int(subid.copy())
                length_aug1 = length - trnc_len
                wvf_aug1 = wvf[:length_aug1]
                
                if input.shape[0] == 1:
                    input_aug1 = resample(input[0][:-trnc_len], 1000)
                    input_aug1 = [butterworth_filter(input_aug1, 30, 1000, 4, type_='low')]
                elif input.shape[0] == 3:
                    input_aug1 = [
                        resample(input[0][:-trnc_len], 1000), 
                        resample(input[1][:-trnc_len], 1000), 
                        resample(input[2][:-trnc_len], 1000),
                        ]
                    input_aug1 = [
                        butterworth_filter(input_aug1[0], 30, 1000, 4, type_='low'), 
                        butterworth_filter(input_aug1[1], 30, 1000, 4, type_='low'), 
                        butterworth_filter(input_aug1[2], 30, 1000, 4, type_='low'),
                        ]
                    input_aug1 = list(input_aug1)
                else:
                    input_aug1 = resample(input[:-trnc_len], 1000)
                    input_aug1 = butterworth_filter(input_aug1, 30, 1000, 4, type_='low')    
                    
                inputs_aug.append(input_aug1)
                outputs_aug.append(output_aug1)
                indices_aug.append(index_aug1)
                subids_aug.append(subid_aug1)
                lengths_aug.append(length_aug1)
                wvfs_avg.append(wvf_aug1)

        # augment data method 2: adding noise to the waveforms
        if aug_type[1]:
            if random.randint(1, aug_prob[1]) == 1:
                noise = np.random.normal(0, 0.01, input[0].shape)
                output_aug2 = output
                if input.shape[0] == 1:
                    input_aug2 = input[0] + noise
                elif input.shape[0] == 3:
                    input_noise = input[0] + noise
                    input_aug2 = [input_noise, np.gradient(input_noise), np.gradient(np.gradient(input_noise))]
                else:
                    input_aug2 = input + noise
                index_aug2 = index.copy()
                index_aug2[1] = 3
                subid_aug2 = int(subid.copy())
                length_aug2 = length
                wvf_aug2 = wvf[:length_aug2]

                inputs_aug.append(input_aug2)
                outputs_aug.append(output_aug2)
                indices_aug.append(index_aug2)
                subids_aug.append(subid_aug2)
                lengths_aug.append(length_aug2)
                wvfs_avg.append(wvf_aug2)
        
        # augment data method 3: scaling the waveforms
        if aug_type[2]:
            if random.randint(1, aug_prob[2]) == 1:
                scale = random.uniform(0.5, 1.5)
                output_aug3 = output
                if input.shape[0] == 1:
                    input_aug3 = input[0] * scale
                elif input.shape[0] == 3:
                    input_aug3 = [x * scale for x in input]
                else:
                    input_aug3 = input * scale
                index_aug3 = index.copy()
                index_aug3[1] = 4
                subid_aug3 = int(subid.copy())
                length_aug3 = length
                wvf_aug3 = wvf[:length_aug3]

                inputs_aug.append(input_aug3)
                outputs_aug.append(output_aug3)
                indices_aug.append(index_aug3)
                subids_aug.append(subid_aug3)
                lengths_aug.append(length_aug3)
                wvfs_avg.append(wvf_aug3)

        # augment data method 4: time shift waveforms
        if aug_type[3]:
            if random.randint(1, aug_prob[3]) == 1:
                time_shift = random.randint(-35, 35)
                output_aug4 = output + time_shift/1000
                index_aug4 = index.copy()
                index_aug4[1] = 5
                subid_aug4 = int(subid.copy())
                length_aug4 = length
                wvf_aug4 = np.roll(wvf[:length], shift=time_shift)

                if input.shape[0] == 1:
                    input_aln = input[0] - np.linspace(0, input[0][-1] - input[0][0], 1000)
                    input_aug4 = np.roll(input_aln, shift=time_shift)
                elif input.shape[0] == 3:
                    input_aln = input[0] - np.linspace(0, input[0][-1] - input[0][0], 1000)
                    input_aug4 = [
                        np.roll(input_aln, shift=time_shift),
                        np.gradient(np.roll(input_aln, shift=time_shift)),
                        np.gradient(np.gradient(np.roll(input_aln, shift=time_shift))),
                    ]
                else:
                    input_aln = input - np.linspace(0, input[-1] - input[0], 1000)
                    input_aug4 = np.roll(input_aln, shift=time_shift)

                inputs_aug.append(input_aug4)
                outputs_aug.append(output_aug4)
                indices_aug.append(index_aug4)
                subids_aug.append(subid_aug4)
                lengths_aug.append(length_aug4)
                wvfs_avg.append(wvf_aug4)

    # create a new dataset with the augmented data
    inputs_aug = torch.tensor(np.array(inputs_aug), dtype=torch.float32)
    outputs_aug = torch.tensor(np.array(outputs_aug), dtype=torch.float32)
    indices_aug = torch.tensor(np.array(indices_aug), dtype=torch.int64)
    subids_aug = torch.tensor(np.array(subids_aug), dtype=torch.int64)
    lengths_aug = torch.tensor(np.array(lengths_aug), dtype=torch.int64)
    wvfs_aug = torch.nn.utils.rnn.pad_sequence([torch.tensor(x, dtype=torch.float32) for x in wvfs_avg], padding_value=torch.nan, batch_first=True)

    # print the size of the augmented dataset
    print(inputs_aug.size(), outputs_aug.size(), indices_aug.size(), subids_aug.size(), lengths_aug.size(), wvfs_aug.size())

    # compose the augmented dataset
    dataset_aug = TensorDataset(inputs_aug, outputs_aug, indices_aug, subids_aug, lengths_aug, wvfs_aug)

    return dataset_aug

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
                 channels_present: bool = False,
                 train_proportion: float = 1.0, # this flag is used to select a subset of the train population for training
                 train_proportion_seed: int = None,
                 augment_train_data: bool = False,
                 augment_type: list = [False, False, False, False],
                 augment_prob: list = [1, 1, 1, 1],
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
        self.data_split = False
        self.channels_present = channels_present
        self.train_proportion = train_proportion
        self.train_proportion_seed = seed if train_proportion_seed is None else train_proportion_seed
        self.augment_train_data = augment_train_data
        self.augment_type = augment_type
        self.augment_prob = augment_prob
        
    def prepare_data(self):
        # load the dataset
        df = pd.read_pickle(os.path.join(self.data_dir, self.data_fname))
        
        # drop any nan row
        df = df.dropna()
        df['p_ind'] = df['p_ind'].astype(int)
        df['i_ind'] = df['i_ind'].astype(int)
        df['n_ind'] = df['n_ind'].astype(int)
        
        # preprocess the dataset to normalize the waveforms for input to the CNN
        inputs, outputs, indices, subids, lengths, calib_wvfs = data_preprocessing(df)

        # format the inputs data
        if self.channels_present:
            # reshape inputs to be in the format (batch_size, channels, sequence_length)
            inputs = inputs.unsqueeze(1)
        
        return inputs, outputs, indices, subids, lengths, calib_wvfs

    def setup(self, stage=None):
        # Assign train, validation, and test datasets for use in dataloaders
        print('Stage: ', stage)
        if self.data_split is False and (stage == 'fit' or stage is None):
            inputs, outputs, indices, subids, lengths, wvfs = self.prepare_data()
            if self.split_type == 'cardiac_cycle':
                dataset = TensorDataset(inputs, outputs, indices, subids, lengths, wvfs)
                self.train_data, self.val_data, self.test_data = random_split(dataset, [self.size['train'], self.size['val'], self.size['test']], generator=self.generator)
            elif self.split_type == 'subid':
                # split the datasets according to subids to avoid data leakage
                unique_subids = StringDataset(subids.unique())
                train_, val_, test_ = random_split(unique_subids, [self.size['train'], self.size['val'], self.size['test']], generator=self.generator)
                train_subids = unique_subids[train_.indices]
                val_subids = unique_subids[val_.indices]
                test_subids = unique_subids[test_.indices]

                # select a random subset of the train population for training
                if self.train_proportion < 1.0:
                    random.seed(self.train_proportion_seed)
                    idx = random.sample(range(len(train_subids)), int(self.train_proportion*len(train_subids)))
                    train_subids = train_subids[idx]
                
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
                lengths_train = lengths[inds_train]
                wvfs_train = wvfs[inds_train]
                self.train_data = TensorDataset(inputs_train, outputs_train, indices_train, subids_train, lengths_train, wvfs_train)
                if self.augment_train_data:
                    self.train_data = dataset_augmentation(self.train_data, self.augment_type, self.augment_prob)
                
                inputs_val = inputs[inds_val]
                outputs_val = outputs[inds_val]
                indices_val = indices[inds_val]
                subids_val = subids[inds_val]
                lengths_val = lengths[inds_val]
                wvfs_val = wvfs[inds_val]
                self.val_data = TensorDataset(inputs_val, outputs_val, indices_val, subids_val, lengths_val, wvfs_val)

                inputs_test = inputs[inds_test]
                outputs_test = outputs[inds_test]
                indices_test = indices[inds_test]
                subids_test = subids[inds_test]
                lengths_test = lengths[inds_test]
                wvfs_test = wvfs[inds_test]
                self.test_data = TensorDataset(inputs_test, outputs_test, indices_test, subids_test, lengths_test, wvfs_test)
            else:
                print("unknown split type")

            # change split flag
            self.data_split = True
        else:
            print('Data already split.')
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
    
    def return_subid_split(self):
        return self.subid_split
    
class WaveformIndexDataModule_wDerivatives(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = 'data/', 
                 data_fname: str = 'waveform_data.pkl',
                 batch_size: int = 32,
                 size: dict = {'train': 0.8, 'val': 0.1, 'test': 0.1},
                 num_workers: int = 1,
                 seed: int = 3,
                 split_type: str = 'subid',
                 train_proportion: float = 1.0, # this flag is used to select a subset of the train population for training
                 train_proportion_seed: int = None,
                 augment_train_data: bool = False,
                 augment_type: list = [False, False, False, False],
                 augment_prob: list = [1, 1, 1, 1],
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
        self.data_split = False
        self.train_proportion = train_proportion
        self.train_proportion_seed = seed if train_proportion_seed is None else train_proportion_seed
        self.augment_train_data = augment_train_data
        self.augment_type = augment_type
        self.augment_prob = augment_prob
        
    def prepare_data(self):
        # load the dataset
        df = pd.read_pickle(os.path.join(self.data_dir, self.data_fname))

        # drop any nan row
        df = df.dropna()
        df['p_ind'] = df['p_ind'].astype(int)
        df['i_ind'] = df['i_ind'].astype(int)
        df['n_ind'] = df['n_ind'].astype(int)
        
        # preprocess the dataset to normalize the waveforms for input to the CNN
        inputs, outputs, indices, subids, lengths, calib_wvfs = data_preprocessing_with_derivatives(df)
        
        return inputs, outputs, indices, subids, lengths, calib_wvfs

    def setup(self, stage=None):
        # Assign train, validation, and test datasets for use in dataloaders
        print('Stage: ', stage)
        if self.data_split is False and (stage == 'fit' or stage is None):
            inputs, outputs, indices, subids, lengths, wvfs = self.prepare_data()
            if self.split_type == 'cardiac_cycle':
                dataset = TensorDataset(inputs, outputs, indices, subids, lengths, wvfs)
                self.train_data, self.val_data, self.test_data = random_split(dataset, [self.size['train'], self.size['val'], self.size['test']], generator=self.generator)
            elif self.split_type == 'subid':
                # split the datasets according to subids to avoid data leakage
                unique_subids = StringDataset(subids.unique())
                train_, val_, test_ = random_split(unique_subids, [self.size['train'], self.size['val'], self.size['test']], generator=self.generator)
                train_subids = unique_subids[train_.indices]
                val_subids = unique_subids[val_.indices]
                test_subids = unique_subids[test_.indices]

                # select a random subset of the train population for training
                if self.train_proportion < 1.0:
                    random.seed(self.train_proportion_seed)
                    idx = random.sample(range(len(train_subids)), int(self.train_proportion*len(train_subids)))
                    train_subids = train_subids[idx]
                
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
                lengths_train = lengths[inds_train]
                wvfs_train = wvfs[inds_train]
                self.train_data = TensorDataset(inputs_train, outputs_train, indices_train, subids_train, lengths_train, wvfs_train)
                if self.augment_train_data:
                    self.train_data = dataset_augmentation(self.train_data, self.augment_type, self.augment_prob)
                
                inputs_val = inputs[inds_val]
                outputs_val = outputs[inds_val]
                indices_val = indices[inds_val]
                subids_val = subids[inds_val]
                lengths_val = lengths[inds_val]
                wvfs_val = wvfs[inds_val]
                self.val_data = TensorDataset(inputs_val, outputs_val, indices_val, subids_val, lengths_val, wvfs_val)

                inputs_test = inputs[inds_test]
                outputs_test = outputs[inds_test]
                indices_test = indices[inds_test]
                subids_test = subids[inds_test]
                lengths_test = lengths[inds_test]
                wvfs_test = wvfs[inds_test]
                self.test_data = TensorDataset(inputs_test, outputs_test, indices_test, subids_test, lengths_test, wvfs_test)
            else:
                print("unknown split type")

            # change split flag
            self.data_split = True
        else:
            print('Data already split.')
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
    
    def return_subid_split(self):
        return self.subid_split