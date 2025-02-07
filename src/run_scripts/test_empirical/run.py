import sys
import argparse

sys.path.append('../../')
from models.empirical.runner import Runner

def main():
    parser = argparse.ArgumentParser(description='Run the empirical model on the given dataset.')
    parser.add_argument('--project_name', type=str, default='fiducial-points', help='Name of the project.')
    parser.add_argument('--model_name', type=str, default='empirical', help='Name of the model.')
    # data hyperparameters
    parser.add_argument('--data_dir', type=str, default='../../../data/', help='Path to the data directory.')
    parser.add_argument('--data_fname', type=str, default='wvfIndexData_AO_vrfd_cnv.pkl', help='Name of the data file.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloaders.')
    parser.add_argument('--size', type=dict, default={'train': 0.7, 'val': 0.1, 'test': 0.2}, help='Size of the training, validation, and test sets.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--split_type', type=str, default='subid', help='Type of data split.')
    parser.add_argument('--num_workers', type=int, default=9, help='Number of workers for the dataloaders.')
    parser.add_argument('--train_proportion', type=float, default=1.0, help='The percentage of training set to use.')
    # model hyperparameters
    parser.add_argument('--special_test', type=str, default=None, help='Special test to run.')
    # training hyperparameters
    # other hyperparameters
    
    # build the experiment configuration
    args = parser.parse_args()
    config = vars(args)
    Runner(**config)

if __name__ == '__main__':
    main()