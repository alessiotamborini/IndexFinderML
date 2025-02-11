import torch
import torch.nn as nn

class mcCNN(nn.Module):
    """
    A convolutional neural network model for the prediction 
    of the index of a normalized waveform. This model takes 
    multiple channels as input representing the waveform, 
    its first and second derivative. 
    """

    def __init__(self,
                 input_dim: int = 3,
                 output_dim: int = 2,
                 layer_dims: list = [16, 32, 64],
                 kernel_dim: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dropout: float = 0.35,
                 activation_type: str = 'relu',
        ):
        super(mcCNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_dims = layer_dims
        self.kernel_dim = kernel_dim
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.activation_type = activation_type

        # Define the activation function
        if self.activation_type == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError('Activation type not supported.')
        
        # convolutional block 1
        self.conv1 = nn.Conv1d(input_dim, layer_dims[0], kernel_size=kernel_dim, stride=stride, padding=padding)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # convolutional block 2
        self.conv2 = nn.Conv1d(layer_dims[0], layer_dims[1], kernel_size=kernel_dim, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # convolutional block 3
        self.conv3 = nn.Conv1d(layer_dims[1], layer_dims[2], kernel_size=kernel_dim, stride=stride, padding=padding)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # fully connected layers
        self.fc1 = nn.Linear(layer_dims[2] * (1000 // (2**3)), 128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):

        # Convolutional block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Convolutional block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Convolutional block 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)

        return x
