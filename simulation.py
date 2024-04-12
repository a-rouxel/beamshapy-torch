import torch
from torch import nn
from units import *

class Simulation(nn.Module):
    def __init__(self, config_dict= None):
        super().__init__()

        self.config_dict = config_dict

        self.grid_size = self.config_dict["grid size"]*mm
        self.delta_x_in= self.config_dict["grid sampling"]*um

        self.generate_input_grid()
        
    def generate_input_grid(self):
        x = torch.arange(-self.grid_size/2, self.grid_size/2, self.delta_x_in)
        y = torch.arange(-self.grid_size/2, self.grid_size/2, self.delta_x_in)
        self.XY_grid = torch.meshgrid(x, y)
        return self.XY_grid
