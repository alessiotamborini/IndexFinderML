import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A multi-layer perceptron model for the prediction
    of the index of a normalized waveform.
    """

    def __init__(self,
                 input_dim: int = 1000,
                 output_dim: int = 3,
                 layer_dims: list = [128, 64, 32],
                 dropout: float = 0.35,
                 activation_type: str = 'relu',
        ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_dims = layer_dims
        self.dropout = dropout
        self.activation_type = activation_type

        # Define the activation function
        if self.activation_type == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError('Activation type not supported.')
        
        # fully connected layers
        self.fc1 = nn.Linear(input_dim, layer_dims[0])
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(layer_dims[0], layer_dims[1])
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(layer_dims[1], layer_dims[2])
        self.relu_fc3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(layer_dims[2], output_dim)

    def forward(self, x):


        # Fully connected layers
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.dropout3(x)

        x = self.fc4(x)

        return x
