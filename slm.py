import torch
from torch import nn

class SLM(nn.Module):

    def __init__(self, config_dict= None, XY_grid= None,initial_phase=None, initial_amplitude=None):
        super().__init__()

        self.config_dict = config_dict
        self.XY_grid = XY_grid
        self.initial_phase = initial_phase
        self.initial_amplitude = initial_amplitude

        self.generate_initial_phase()
        self.generate_initial_amplitude()
    
    def generate_initial_phase(self):

        if self.initial_phase is not None:
            self.phase = nn.Parameter(torch.tensor(self.initial_phase))
        
        elif self.config_dict["initial phase"] == "random":
            self.phase = nn.Parameter(torch.rand(self.XY_grid[0].shape))
        
        elif self.config_dict["initial phase"] == "zero":
            self.phase = nn.Parameter(torch.zeros(self.XY_grid[0].shape))

        elif self.config_dict["initial phase"] == "custom":
            raise("Custom phase should be implemented")
    
    def generate_initial_amplitude(self):

        if self.initial_amplitude is not None:
            self.amplitude = nn.Parameter(torch.tensor(self.initial_amplitude))
            
        elif self.config_dict["initial amplitude"] == "ones":
            self.amplitude = nn.Parameter(torch.ones(self.XY_grid[0].shape))

        elif self.config_dict["initial amplitude"] == "custom":
            raise("Custom amplitude should be implemented")