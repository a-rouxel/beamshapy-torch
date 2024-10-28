from torch import nn
import torch
from units import *
from helpers import load_yaml_config
import matplotlib.pyplot as plt
from electric_field import ElectricField
import json
from math import factorial
from simulation import Simulation
import numpy as np
from PIL import Image


class Source(nn.Module):
    def __init__(self, config_dict= None,XY_grid= None):
        super().__init__()
        self.config_dict = config_dict
        self.type = self.config_dict["type"]
        self.num_gaussians = self.config_dict["num gaussian"]
        self.wavelength = self.config_dict["wavelength"]*nm
        self.initial_power = self.config_dict["power"]
        self.init_amp = 1
        self.XY_grid = XY_grid

        self.amplitudes = nn.Parameter(torch.ones(self.num_gaussians) / (self.num_gaussians) * self.init_amp)
        self.amplitudes.requires_grad = False
        self.means = nn.Parameter(torch.zeros(self.num_gaussians, 2) * mm)
        self.means.requires_grad = False
        self.sigmas = nn.Parameter(torch.ones(self.num_gaussians) * 1.2* mm)
        self.sigmas.requires_grad = False
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
            sigma_squared = self.sigmas[i] ** 2
            exponent = - (diff_x + diff_y) / sigma_squared
            # two_pi_tensor = torch.tensor(2 * torch.pi, dtype=torch.float32, device=x.device)
            # prefac = 1 / (self.sigma[i] * torch.sqrt(two_pi_tensor))
            amplitude_idx = torch.exp(exponent)*self.amplitudes[i]
            amplitude += amplitude_idx
        # intensity = amplitude ** 2
        # current_power = intensity.sum()
        # amplitude *= (self.initial_power / current_power)
        # amplitude = torch.sqrt(intensity*(self.initial_power / current_power))
        return amplitude
    
    def generate_electric_field(self):

        if self.type == "gaussian mixture":
            amplitude = self.gaussian_mixture()
            phase = self.phase
            self.field = ElectricField(amplitude, phase,self.XY_grid)
        else:
            raise NotImplementedError
        
        return self.field
    

class ZernikePolynomials:
    @staticmethod
    def zernike_radial(n, m, rho):
        R = torch.zeros_like(rho)
        for k in range((n - abs(m)) // 2 + 1):
            R += ((-1) ** k * factorial(n - k) /
                  (factorial(k) *
                   factorial((n + abs(m)) // 2 - k) *
                   factorial((n - abs(m)) // 2 - k))) * rho ** (n - 2 * k)
        return R

    @staticmethod
    def generate_zernike_phase_map(coefficients, rho, theta):
        phase_map = torch.zeros_like(rho)
        for (n, m), coeff in coefficients.items():
            phase_map += coeff * ZernikePolynomials.zernike_radial(n, abs(m), rho) * torch.cos(m * theta)
        return phase_map

class GaussianMixtureSource(nn.Module):
    def __init__(self, config_dict=None, XY_grid=None):
        super().__init__()
        self.config_dict = config_dict
        self.num_gaussians = self.config_dict["num gaussian"]
        self.wavelength = self.config_dict["wavelength"]*nm  # Add wavelength
        self.initial_power = self.config_dict["power"]       # Add power
        self.init_amp = 1                                    # Add init_amp
        self.XY_grid = XY_grid
        self.centroid = None  # Add this line

        # Load parameters from JSON if specified in config
        params_file = self.config_dict.get("parameters_file", None)
        if params_file:
            self.load_parameters(params_file)
        else:
            # Default initialization as before
            self.amplitudes = nn.Parameter(torch.ones(self.num_gaussians) / self.num_gaussians * self.init_amp)
            self.means = nn.Parameter(torch.zeros(self.num_gaussians, 2) * mm)
            self.sigmas = nn.Parameter(torch.ones(self.num_gaussians, 2) * mm)
            self.offset = nn.Parameter(torch.zeros(1))
        
        # Generate Zernike phase map
        x, y = self.XY_grid
        # Normalize coordinates to [-1, 1]
        x_norm = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        y_norm = 2 * (y - y.min()) / (y.max() - y.min()) - 1
        rho = torch.sqrt(x_norm**2 + y_norm**2)
        theta = torch.atan2(y_norm, x_norm)
        
        # Get Zernike coefficients from config and convert string keys to tuples
        zernike_coeffs_raw = self.config_dict.get("zernike_coefficients", {})
        zernike_coefficients = {}
        for key, value in zernike_coeffs_raw.items():
            # Convert string key '(n,m)' to tuple (n,m)
            n, m = map(int, key.strip('()').split(','))
            zernike_coefficients[(n, m)] = value
        
        self.phase = nn.Parameter(
            ZernikePolynomials.generate_zernike_phase_map(zernike_coefficients, rho, theta),
            requires_grad=False
        )
        self.generate_electric_field()

    def load_parameters(self, json_file):
        with open(json_file, 'r') as f:
            params = json.load(f)
            
        pixel_size = params.get('pixel_size', 3.45) * um
        
        # Load centroid information
        if 'centroid' in params:
            self.centroid = {
                'x': params['centroid']['pixels'][0] * pixel_size,
                'y': params['centroid']['pixels'][1] * pixel_size
            }
        else:
            raise ValueError("Centroid information missing in parameters file")
        
        # Convert parameters to tensors with proper units and adjust means relative to centroid
        self.amplitudes = nn.Parameter(torch.tensor(params['amplitudes']))
        means_tensor = torch.tensor(params['means']) * pixel_size
        
        # Store the original means adjusted by centroid
        self.means = nn.Parameter(means_tensor)
        self.sigmas = nn.Parameter(torch.tensor(params['sigmas']) * pixel_size)
        self.offset = nn.Parameter(torch.tensor([params['offset']]))
        
        # Load Zernike coefficients if present
        if 'zernike_coefficients' in params:
            zernike_coefficients = {
                tuple(map(int, key.strip('()').split(','))): value 
                for key, value in params['zernike_coefficients'].items()
            }
            # Regenerate phase map with loaded coefficients
            x, y = self.XY_grid
            x_norm = 2 * (x - x.min()) / (x.max() - x.min()) - 1
            y_norm = 2 * (y - y.min()) / (y.max() - y.min()) - 1
            rho = torch.sqrt(x_norm**2 + y_norm**2)
            theta = torch.atan2(y_norm, x_norm)
            self.phase = nn.Parameter(
                ZernikePolynomials.generate_zernike_phase_map(zernike_coefficients, rho, theta),
                requires_grad=False
            )
        
        # Update num_gaussians if different
        loaded_num_gaussians = len(params['amplitudes'])
        if loaded_num_gaussians != self.num_gaussians:
            print(f"Warning: Loaded {loaded_num_gaussians} gaussians, but config specified {self.num_gaussians}")
            self.num_gaussians = loaded_num_gaussians

    def gaussian_mixture(self):
        x, y = self.XY_grid
        result = torch.zeros_like(x)
        
        for i in range(self.num_gaussians):
            # Adjust means by subtracting centroid
            means_x = (self.means[i, 0] - self.centroid['x']).expand_as(x)
            means_y = (self.means[i, 1] - self.centroid['y']).expand_as(y)
            
            gaussian = self.amplitudes[i] * torch.exp(
                -(((x - means_x)**2 / (self.sigmas[i, 0]**2)) + 
                  ((y - means_y)**2 / (self.sigmas[i, 1]**2)))
            )
            result = result + gaussian
            
        return result

    def generate_electric_field(self):
        """
        Generates the electric field based on the gaussian mixture with Zernike phase.
        Returns an ElectricField object with the calculated amplitude and phase.
        """
        amplitude = self.gaussian_mixture()
        
        # Normalize the amplitude based on initial power if specified
        if self.initial_power is not None:
            intensity = amplitude ** 2
            current_power = intensity.sum()
            amplitude = torch.sqrt(intensity * (self.initial_power / current_power))
        
        # Create complex field using amplitude and Zernike phase
        self.field = ElectricField(amplitude, self.phase, self.XY_grid)
        return self.field
    

if __name__ == "__main__":
    file_path = "./input_beam/vanilla_collimator_fixed2.png"

    # Load and convert image to tensor first to get its dimensions
    image = Image.open(file_path).convert('L')
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32)
    img_height, img_width = image_tensor.shape

    config_simulation = load_yaml_config("./configs/simulation_ASM.yml")
    simulation = Simulation(config_dict=config_simulation)

    config_dict = load_yaml_config("./configs/source_gaussian_mixture.yml")
    config_dict_source = load_yaml_config("./configs/source.yml")

    source = GaussianMixtureSource(config_dict=config_dict, XY_grid=simulation.XY_grid)
    source_init = Source(config_dict=config_dict_source, XY_grid=simulation.XY_grid)

    # Get source intensities
    source_intensity = torch.abs(source.field.amplitude.detach())**2
    source_init_intensity = torch.abs(source_init.field.amplitude.detach())**2

    # Calculate center points for cropping
    sh, sw = source_intensity.shape
    start_h = (sh - img_height) // 2
    start_w = (sw - img_width) // 2

    # Crop the source intensities to match image dimensions
    source_intensity_cropped = source_intensity[start_h:start_h+img_height, start_w:start_w+img_width]
    source_init_intensity_cropped = source_init_intensity[start_h:start_h+img_height, start_w:start_w+img_width]

    # Create a single figure with three vertically stacked subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

    # Plot cropped source intensity
    im1 = ax1.imshow(source_intensity_cropped)
    plt.colorbar(im1, ax=ax1, label='Intensity')
    ax1.set_title('Gaussian Mixture Source Intensity')

    # Plot cropped source_init intensity
    im2 = ax2.imshow(source_init_intensity_cropped)
    plt.colorbar(im2, ax=ax2, label='Intensity')
    ax2.set_title('Initial Source Intensity')

    # Plot image intensity
    im3 = ax3.imshow(image_tensor)
    plt.colorbar(im3, ax=ax3, label='Intensity')
    ax3.set_title('Input Image Intensity')

    plt.tight_layout()
    plt.show()

    cut_source_intensity_cropped = source_intensity_cropped[source_intensity_cropped.shape[0]//2,:]
    cut_source_init_intensity_cropped = source_init_intensity_cropped[source_init_intensity_cropped.shape[0]//2,:]
    cut_image_tensor = image_tensor[454,:]


    plt.figure(figsize=(10, 10))
    plt.plot(cut_source_intensity_cropped/cut_source_intensity_cropped.max())
    plt.plot(cut_source_init_intensity_cropped/cut_source_init_intensity_cropped.max())
    plt.plot(cut_image_tensor/cut_image_tensor.max())
    plt.show()