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
        # Calculate the number of points
        num_points = int(self.grid_size / self.delta_x_in)
        
        # Ensure the number of points is even
        if num_points % 2 != 0:
            num_points += 1
        
        # Recalculate grid_size to ensure it's consistent with the even number of points
        self.grid_size = num_points * self.delta_x_in
        
        # Generate the grid
        x = torch.linspace(-self.grid_size/2, self.grid_size/2, num_points)
        y = torch.linspace(-self.grid_size/2, self.grid_size/2, num_points)
        self.XY_grid = torch.meshgrid(x, y, indexing='ij')
        return self.XY_grid
