from torch import nn
import torch
from units import *
from helpers import load_yaml_config
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
        
class FT_Lens(nn.Module):
    def __init__(self,delta_x_in, XY_input_grid, wavelength):
        super().__init__()
        self.focal_length = nn.Parameter(torch.tensor(285.0* mm) )
        self.delta_x_in = delta_x_in
        self.XY_input_grid = XY_input_grid
        self.wavelength = wavelength

        self.calculate_delta_x_out()
        self.calculate_output_grid()

    def calculate_delta_x_out(self):
        input_grid_size = (self.XY_input_grid[0].max() - self.XY_input_grid[0].min())
        self.delta_x_out = (self.wavelength * self.focal_length / input_grid_size).detach()


        return self.delta_x_out

    def calculate_output_grid(self):
        Nx = self.XY_input_grid[0].shape[0]
        Ny = self.XY_input_grid[1].shape[1]
        x = torch.arange(-Nx / 2, Nx / 2, 1) * self.delta_x_out
        y = torch.arange(-Ny / 2, Ny / 2, 1) * self.delta_x_out
        self.XY_output_grid = torch.meshgrid(x, y)

        return self.XY_output_grid
    
    
    def forward(self, input_field, norm='ortho', pad=False, flag_ifft=False):
        # Get the initial shape (used later for transforming 6D to 4D if needed)

        input_field = input_field.field
        tmp_shape = input_field.shape

        # Save Size for later crop
        Nx_old = int(input_field.shape[-2])
        Ny_old = int(input_field.shape[-1])
        
        # Pad the image for avoiding convolution artifacts
        if pad:
            pad_scale = 1
            pad_nx = int(pad_scale * Nx_old / 2)
            pad_ny = int(pad_scale * Ny_old / 2)
            input_field = torch.nn.functional.pad(input_field, (pad_nx, pad_nx, pad_ny, pad_ny), mode='constant', value=0)
        
        # Select the appropriate Fourier Transform function based on the flag
        if flag_ifft:
            myfft = torch.fft.ifft2
            my_fftshift = torch.fft.ifftshift
        else:
            myfft = torch.fft.fft2
            my_fftshift = torch.fft.fftshift

        # Compute the Fourier Transform
        out =  my_fftshift(myfft(my_fftshift(input_field, dim=(-2, -1)), dim=(-2, -1), norm=norm), dim=(-2, -1))
        
        # Unpad if padding was used
        if pad:
            out = out[..., pad_ny:-pad_ny, pad_nx:-pad_nx]


        output_field = ElectricField(torch.abs(out), torch.angle(out), self.XY_output_grid)

        return output_field
    


      
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
        self.means = nn.Parameter(torch.zeros(self.num_gaussians, 2) * mm)
        self.sigma = nn.Parameter(torch.tensor([1 / 2] * self.num_gaussians) * mm)
        self.phase = nn.Parameter(torch.zeros(XY_grid[0].shape))  # Random phase for each Gaussian

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
            exponent = -0.5 * (diff_x + diff_y) / sigma_squared
            two_pi_tensor = torch.tensor(2 * torch.pi, dtype=torch.float32, device=x.device)
            prefac = 1 / (self.sigma[i] * torch.sqrt(two_pi_tensor))
            amplitude += prefac * torch.exp(exponent)
        
        current_power = amplitude.sum()
        amplitude *= (self.initial_power / current_power)
        return amplitude
    
    def generate_electric_field(self):

        if self.type == "gaussian mixture":
            amplitude = self.gaussian_mixture()
            phase = self.phase
            self.field = ElectricField(amplitude, phase,self.XY_grid)
        else:
            raise NotImplementedError
        
        return self.field
    
if __name__ == "__main__":

    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling


    config_source = load_yaml_config("./configs/source.yml")
    config_slm = load_yaml_config("./configs/SLM.yml")
    config_simulation = load_yaml_config("./configs/simulation.yml")

    simulation = Simulation(config_dict=config_simulation)

    source = Source(config_dict= config_source, 
                    XY_grid= simulation.XY_grid)
     
    slm = SLM(config_dict= config_slm,
              XY_grid= simulation.XY_grid)
    

    ft_lens = FT_Lens(simulation.delta_x_in, simulation.XY_grid,source.wavelength)

    out_field = ft_lens(source.field, pad=False, flag_ifft=False)

    # out_field.show_field()

    profiler.disable()  # Stop profiling
    stats = pstats.Stats(profiler).sort_stats('cumtime')  # Sort stats by cumulative time
    stats.print_stats(20)
