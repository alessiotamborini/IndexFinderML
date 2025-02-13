import sys

import numpy as np
import torch

from matplotlib import pyplot as plt
from scipy.signal import resample

sys.path.append('../')
from models.mcCNN.mcCNN_pytorch import mcCNN
from utils.butterworth_filter import butterworth_filter

class PimPredictor:
    def __init__(self, model_path):
        super(PimPredictor, self).__init__()
        self.load_pretrained_model(model_path)

    def predict(self, x, dt=1):
        """ 
        Predict the pulse index of a waveform. 
        
        Input:
        -----
        x : (array : float) waveform data
        dt : (float) time step between samples in milliseconds

        Output:
        -----
        output_t : (array : int) pulse indices of inflection and dicrotic notch in units of milliseconds
        """

        # input formatting
        x, sig_len = self.input_formatting(x, dt)

        with torch.no_grad():
            output = self.model(x)
        
        # output formatting
        output = output[0].cpu().detach().numpy()
        output_t = (output*sig_len).astype(int) # bring back to time units
        
        return output_t
    
    def load_pretrained_model(self, model_path):
        """
        Load a pretrained model.

        Args:
            model_path: The path to the pretrained model file.

        Returns:
            model: The loaded model.
        """
        checkpoint = torch.load(model_path, weights_only=False)
        model_hyperparameters = checkpoint['model_hyperparameters']
        model_checkpoint = checkpoint['model_state_dict']

        # filter out unnecessary hyperparameters and initialize the model
        params = ['input_dim','output_dim','layer_dims','kernel_dim','stride','padding','dropout','activation_type']
        model_params = {k: model_hyperparameters[k] for k in params if k in model_hyperparameters.keys()}
        self.model = mcCNN(**model_params)

        # fix state_dict prefix and load the model
        model_checkpoint = {k.replace('model.',''): v for k,v in model_checkpoint.items()}
        self.model.load_state_dict(model_checkpoint)
        self.model.eval()
        
    def input_formatting(self, x, dt):
        """ checks input format to comply with Pulse Indexer Model requirements """

        # ensure input is 1D
        if len(x.shape) != 1:
            x = x[0]
        
        # ensure input length is 1000
        sig_len = len(x)
        if len(x) != 1000:
            x = resample(x, 1000)
        x = butterworth_filter(x, 30, int(1000/dt), 4, 'low')  # filter out high frequency noise

        # ensure we have the correct number of channels
        if x.shape[0] != 3:
            dx = np.gradient(x)
            ddx = np.gradient(dx)
            x = np.vstack([x, dx, ddx])
        
        # ensure input is standardized (zero mean, unit variance)
        if any(np.abs(x.mean(axis=1)) > 1e-5) or any(np.abs(x.std(axis=1)) - 1 > 1e-5):
            x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
        
        # ensure final shape is (1, 3, 1000)
        if x.shape != (1, 3, 1000):
            x = x.reshape(1, 3, 1000)
        
        return torch.tensor(x, dtype=torch.float32), sig_len

        

    
    