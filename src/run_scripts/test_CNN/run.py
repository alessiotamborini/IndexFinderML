import sys
import argparse

sys.path.append('../../')
from models.CNN.runner import Runner
from utils.dict_combiner import dict_combiner

def main():
    parser = argparse.ArgumentParser(description='Run the CNN model on the given dataset.')
    parser.add_argument('--project_name', type=str, default='fiducial-points', help='Name of the project.')
    parser.add_argument('--model_name', type=str, default='CNN', help='Name of the model.')
    # data hyperparameters
    parser.add_argument('--data_dir', type=str, default='../../../data/', help='Path to the data directory.')
    parser.add_argument('--data_fname', type=str, default='wvfIndexData_AO_vrfd_cnv.pkl', help='Name of the data file.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloaders.')
    parser.add_argument('--size', type=dict, default={'train': 0.7, 'val': 0.1, 'test': 0.2}, help='Size of the training, validation, and test sets.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--split_type', type=str, default='subid', help='Type of data split.')
    parser.add_argument('--num_workers', type=int, default=9, help='Number of workers for the dataloaders.')
    parser.add_argument('--channels_present', type=bool, default=True, help='Whether the data has channels.')
    parser.add_argument('--train_proportion', type=float, default=1.0, help='The percentage of training set to use.')
    # model hyperparameters
    parser.add_argument('--input_dim', type=int, default=1, help='Input dimension of the data.')
    parser.add_argument('--output_dim', type=int, default=3, help='Output dimension of the data.')
    parser.add_argument('--layer_dims', type=list, default=[8, 16, 32], help='Dimensions of the layers in the model.')
    parser.add_argument('--kernel_dim', type=int, default=3, help='Kernel dimension for the convolutional layers.')
    parser.add_argument('--stride', type=int, default=1, help='Stride for the convolutional layers.')
    parser.add_argument('--padding', type=int, default=1, help='Padding for the convolutional layers.')
    parser.add_argument('--dropout', type=float, default=0.35, help='Dropout rate for the model.')
    parser.add_argument('--activation_type', type=str, default='relu', help='Type of activation function to use.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--monitor_metric', type=str, default='loss/val/mse', help='Metric to monitor for early stopping.')
    parser.add_argument('--special_test', type=str, default=None, help='Special test to run.')
    # training hyperparameters
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs to train the model.')
    parser.add_argument('--gradient_clip_val', type=float, default=10.0, help='Value for gradient clipping.')
    parser.add_argument('--gradient_clip_algorithm', type=str, default='value', help='Algorithm for gradient clipping.')
    parser.add_argument('--deterministic', type=bool, default=True, help='Whether to set the random seed for deterministic training.')
    parser.add_argument('--devices', type=str, default='auto', help='Devices to use for training.')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Accelerator to use for training.')
    # other hyperparameters
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    
    # build the experiment configuration
    args = parser.parse_args()
    config = vars(args)
    Runner(**config)

if __name__ == '__main__':
    main()