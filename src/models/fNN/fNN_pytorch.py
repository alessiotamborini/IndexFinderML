import torch
import torch.nn as nn

class fNN(nn.Module):
    """
    A fourier-based neural network model for the prediction
    of the index of a normalized waveform.
    """

    def __init__(self,
                 input_modes: int = 20,  # number of modes
                 output_dim: int = 3,
                 layer_dims: list = [128, 64, 32],
                 dropout: float = 0.35,
                 activation_type: str = 'relu',
        ):
        super(fNN, self).__init__()

        self.input_modes = input_modes
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
        self.fc1 = nn.Linear(input_modes*2-1, layer_dims[0])
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

        # Step 1: Convert the signal from time to frequency using rfft
        x_freq = torch.fft.rfft(x, norm='forward')

        # Step 2: Truncate to a limited number of modes
        x_freq = x_freq[:, :self.input_modes]

        # step 3: compose a torch with the real and imag parts
        x_fft = torch.concatenate((x_freq.real, x_freq.imag[:, 1:]), dim=-1)

        # Fully connected layers
        x_fft = self.fc1(x_fft)
        x_fft = self.relu_fc1(x_fft)
        x_fft = self.dropout1(x_fft)

        x_fft = self.fc2(x_fft)
        x_fft = self.relu_fc2(x_fft)
        x_fft = self.dropout2(x_fft)

        x_fft = self.fc3(x_fft)
        x_fft = self.relu_fc3(x_fft)
        x_fft = self.dropout3(x_fft)

        x_fft = self.fc4(x_fft)

        return x_fft
