from torch import nn
import torch
from units import *
from helpers import load_yaml_config
import matplotlib.pyplot as plt
from electric_field import ElectricField



class Source(nn.Module):
    def __init__(self, config_dict= None,XY_grid= None):
        super().__init__()
        self.config_dict = config_dict
        self.type = self.config_dict["type"]
        self.num_gaussians = self.config_dict["num gaussian"]
        self.wavelength = self.config_dict["wavelength"]*nm
        self.initial_power = self.config_dict["power"]
        self.XY_grid = XY_grid

        self.weights = nn.Parameter(torch.tensor([1.0] * self.num_gaussians))
        self.weights.requires_grad = False
        self.means = nn.Parameter(torch.zeros(self.num_gaussians, 2) * mm)
        self.means.requires_grad = False
        self.sigma = nn.Parameter(torch.tensor([1.3] * self.num_gaussians) * mm)
        self.phase = nn.Parameter(torch.zeros(XY_grid[0].shape))
        self.phase.requires_grad = False

        self.generate_electric_field()

    def gaussian_mixture(self):

        x, y = self.XY_grid
        amplitude = torch.zeros_like(x)
        
        for i in range(self.num_gaussians):
            means_x = self.means[i, 0].expand_as(x)
            means_y = self.means[i, 1].expand_as(y)
            diff_x = (x - means_x) ** 2
            diff_y = (y - means_y) ** 2
            sigma_squared = self.sigma[i] ** 2
            exponent = - (diff_x + diff_y) / sigma_squared
            two_pi_tensor = torch.tensor(2 * torch.pi, dtype=torch.float32, device=x.device)
            # prefac = 1 / (self.sigma[i] * torch.sqrt(two_pi_tensor))
            amplitude += torch.exp(exponent)

        intensity = amplitude ** 2
        current_power = intensity.sum()
        # amplitude *= (self.initial_power / current_power)
        amplitude = torch.sqrt(intensity*(self.initial_power / current_power))
        return amplitude
    
    def generate_electric_field(self):

        if self.type == "gaussian mixture":
            amplitude = self.gaussian_mixture()
            phase = self.phase
            self.field = ElectricField(amplitude, phase,self.XY_grid)
        else:
            raise NotImplementedError
        
        return self.field
    
