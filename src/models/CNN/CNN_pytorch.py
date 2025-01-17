import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    A convolutional neural network model.
    """

    def __init__(self,
                 input_dim: int = 1000,
                 output_dim: int = 2,
                 layer_dims: list = [256, 128, 64],
                 kernel_dim: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dropout: float = 0.35,
                 activation_type: str = 'relu',
        ):
        super(CNN, self).__init__()

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
        elif self.activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif self.activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation type not supported.')
        
        # Define the convolutional layers
        self.conv1 = nn.Conv1d(input_dim, layer_dims[0], kernel_size=kernel_dim, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(layer_dims[0])
        self.conv2 = nn.Conv1d(layer_dims[0], layer_dims[1], kernel_size=kernel_dim,stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(layer_dims[1])
        self.conv3 = nn.Conv1d(layer_dims[1], layer_dims[2], kernel_size=kernel_dim,stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm1d(layer_dims[2])

        # Define the fully connected layers
        self.fc1 = nn.Linear(layer_dims[2], output_dim)

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc1(x)

        return x