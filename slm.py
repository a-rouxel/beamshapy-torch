import torch
from torch import nn
from cost_functions import generalized_sigmoid_function
import matplotlib.pyplot as plt

class SLM(nn.Module):

    def __init__(self, config_dict=None, XY_grid=None, initial_phase=None, initial_amplitude=None, device=None):
        super().__init__()

        # Set the device (CUDA if available, else CPU)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config_dict = config_dict
        self.XY_grid = XY_grid
        self.initial_phase = initial_phase
        self.initial_amplitude = initial_amplitude

        self.generate_initial_phase()
        self.generate_initial_amplitude()

        # Move the entire model to the specified device
        self.to(self.device)

    def generate_initial_phase(self):
        if self.initial_phase is not None:
            self.phase_parameters = nn.Parameter(torch.tensor(self.initial_phase, device=self.device))
        
        elif self.config_dict["initial phase"] == "random":
            self.phase_parameters = nn.Parameter(2*torch.rand(self.XY_grid[0].shape, device=self.device)-1)

        elif self.config_dict["initial phase"] == "zero":
            self.phase_parameters = nn.Parameter(torch.zeros(self.XY_grid[0].shape, device=self.device))

        elif self.config_dict["initial phase"] == "custom":
            raise NotImplementedError("Custom phase should be implemented")
    
    def generate_initial_amplitude(self):
        if self.initial_amplitude is not None:
            self.amplitude = nn.Parameter(torch.tensor(self.initial_amplitude, device=self.device))
            
        elif self.config_dict["initial amplitude"] == "ones":
            self.amplitude = nn.Parameter(torch.ones(self.XY_grid[0].shape, device=self.device))

        elif self.config_dict["initial amplitude"] == "custom":
            raise NotImplementedError("Custom amplitude should be implemented")

    def apply_phase_modulation(self, input_field,beta=1.0,mapping=True,parity=0):
          # self.phase = 0.5*torch.pi* (torch.tanh(beta*self.phase_parameters))
        if mapping:
            if parity == 0:
                self.phase = 2*torch.pi*torch.sigmoid(beta*self.phase_parameters)
            else:
                self.phase = torch.pi*torch.sigmoid(beta*self.phase_parameters) - torch.pi/2
        else :
            self.phase = self.phase_parameters


        return torch.abs(input_field) * torch.exp(1j * self.phase)

    def apply_phase_modulation_sigmoid(self, input_field,steepness=20,num_terms=3, spacing=0.5,mode_parity="even"):

        self.phase = generalized_sigmoid_function(self.phase_parameters,
                                                  steepness=steepness,
                                                  num_terms=num_terms,
                                                  spacing=spacing,
                                                  mode_parity=mode_parity)

        return torch.abs(input_field) * torch.exp(1j * self.phase)
    
    def apply_amplitude_modulation(self, input_field):
          amplitude_input = torch.abs(input_field)
          modulated_amplitude = amplitude_input * self.amplitude
          return modulated_amplitude * torch.exp(1j * torch.angle(input_field))
