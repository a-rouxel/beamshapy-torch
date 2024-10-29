import torch
from torch import nn
from cost_functions import generalized_sigmoid_function
import matplotlib.pyplot as plt

class SLM(nn.Module):

    def __init__(self, config_dict=None, XY_grid=None, initial_phase=None, initial_amplitude=None, device=None, downsampling=1):
        super().__init__()

        # Set the device (CUDA if available, else CPU)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config_dict = config_dict
        self.XY_grid = XY_grid
        self.initial_phase = initial_phase
        self.initial_amplitude = initial_amplitude
        self.downsampling = downsampling

        # Calculate downsampled dimensions
        if XY_grid is not None:
            self.full_shape = XY_grid[0].shape
            self.downsampled_shape = (
                self.full_shape[0] // downsampling,
                self.full_shape[1] // downsampling
            )
        else:
            self.full_shape = None
            self.downsampled_shape = None

        self.generate_initial_phase()
        self.generate_initial_amplitude()

        # Move the entire model to the specified device
        self.to(self.device)

    def generate_initial_phase(self):
        if self.initial_phase is not None:
            # Downsample the initial phase if provided
            if self.downsampling > 1:
                downsampled_phase = self.initial_phase[::self.downsampling, ::self.downsampling]
                self.phase_parameters = nn.Parameter(torch.tensor(downsampled_phase, device=self.device))
            else:
                self.phase_parameters = nn.Parameter(torch.tensor(self.initial_phase, device=self.device))
        
        elif self.config_dict["initial phase"] == "random":
            self.phase_parameters = nn.Parameter(2*torch.rand(self.downsampled_shape, device=self.device)-1)

        elif self.config_dict["initial phase"] == "zero":
            self.phase_parameters = nn.Parameter(torch.zeros(self.downsampled_shape, device=self.device))

        elif self.config_dict["initial phase"] == "custom":
            raise NotImplementedError("Custom phase should be implemented")
    
    def generate_initial_amplitude(self):
        if self.initial_amplitude is not None:
            # Downsample the initial amplitude if provided
            if self.downsampling > 1:
                downsampled_amp = self.initial_amplitude[::self.downsampling, ::self.downsampling]
                self.amplitude = nn.Parameter(torch.tensor(downsampled_amp, device=self.device))
            else:
                self.amplitude = nn.Parameter(torch.tensor(self.initial_amplitude, device=self.device))
            
        elif self.config_dict["initial amplitude"] == "ones":
            self.amplitude = nn.Parameter(torch.ones(self.downsampled_shape, device=self.device))

        elif self.config_dict["initial amplitude"] == "custom":
            raise NotImplementedError("Custom amplitude should be implemented")

    def upsample_parameters(self, downsampled_params):
        """Upsample parameters using repeat to match the original resolution"""
        return downsampled_params.repeat_interleave(self.downsampling, dim=0).repeat_interleave(self.downsampling, dim=1)

    def apply_phase_modulation(self, input_field, beta=1.0, mapping=True, parity=0):
        if mapping:
            if parity == 0:
                downsampled_phase = 2*torch.pi*torch.sigmoid(beta*self.phase_parameters)
            else:
                downsampled_phase = torch.pi*torch.sigmoid(beta*self.phase_parameters) - torch.pi/2
        else:
            downsampled_phase = self.phase_parameters

        # Upsample the phase if downsampling is active
        if self.downsampling > 1:
            self.phase = self.upsample_parameters(downsampled_phase)
        else:
            self.phase = downsampled_phase

        return torch.abs(input_field) * torch.exp(1j * self.phase)

    def apply_phase_modulation_sigmoid(self, input_field, steepness=20, num_terms=3, spacing=0.5):
        # Apply sigmoid to downsampled parameters
        downsampled_phase = generalized_sigmoid_function(
            self.phase_parameters,
            steepness=steepness,
            num_terms=num_terms,
            spacing=spacing
        )

        # Upsample the phase if downsampling is active
        if self.downsampling > 1:
            self.phase = self.upsample_parameters(downsampled_phase)
        else:
            self.phase = downsampled_phase

        return torch.abs(input_field) * torch.exp(1j * self.phase)
    
    def apply_amplitude_modulation(self, input_field):
        # Upsample amplitude if downsampling is active
        if self.downsampling > 1:
            amplitude = self.upsample_parameters(self.amplitude)
        else:
            amplitude = self.amplitude

        amplitude_input = torch.abs(input_field)
        modulated_amplitude = amplitude_input * amplitude
        return modulated_amplitude * torch.exp(1j * torch.angle(input_field))
