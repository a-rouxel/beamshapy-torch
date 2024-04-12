import torch
from torch import nn
import matplotlib.pyplot as plt

class ElectricField(nn.Module):
    def __init__(self, amplitude, phase, XY_grid=None):
        super().__init__()
        self.amplitude = amplitude
        self.phase = phase
        self.XY_grid = XY_grid

        self.compute_field()

    def compute_field(self):
        # Compute the complex electric field: E = A * exp(i * phase)
        self.field = self.amplitude * torch.exp(1j * self.phase)
        return self.field
    
    def show_field(self):

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(self.amplitude.detach().numpy(), extent= [self.XY_grid[0].min(), self.XY_grid[0].max(), self.XY_grid[1].min(), self.XY_grid[1].max()], cmap='hot')
        ax[0].set_xlabel("x [mm]")
        ax[0].set_ylabel("y [mm]")
        ax[0].set_title("Amplitude")

        ax[1].imshow(self.phase.detach().numpy(),extent= [self.XY_grid[0].min(), self.XY_grid[0].max(), self.XY_grid[1].min(), self.XY_grid[1].max()], cmap='viridis')
        ax[1].set_title("Phase")
        ax[1].set_xlabel("x [mm]")
        ax[1].set_ylabel("y [mm]")
        plt.show()
