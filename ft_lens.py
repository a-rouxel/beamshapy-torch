import torch
from torch import nn
from units import *
from electric_field import ElectricField

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def create_binary_fresnel_lens(grid_size, feature_size, wavelength, focal_length, radius=None):
    """
    Creates a binary Fresnel lens phase mask with an optional radius.

    Parameters:
    - grid_size: tuple (num_y, num_x) representing the grid size.
    - feature_size: tuple (dy, dx) representing the pixel size in meters.
    - wavelength: Wavelength of the light in meters.
    - focal_length: Focal length of the lens in meters.
    - radius: Optional radius of the lens in meters. If None, the full grid is used.

    Returns:
    - phase_mask: Tensor of the phase mask with values 0 or π.
    """
    num_y, num_x = grid_size
    dy, dx = feature_size

    # Create coordinate grids
    y = torch.arange(-num_y / 2, num_y / 2) * dy
    x = torch.arange(-num_x / 2, num_x / 2) * dx
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Radial coordinate
    R_squared = X**2 + Y**2

    # Calculate the phase profile
    phase = (np.pi / wavelength / focal_length) * R_squared
    # phase = -1*(R_squared)/(2*wavelength*focal_length)

    # Binary phase mask: 0 or π
    phase_mask = torch.where(torch.remainder(phase, 2 * np.pi) < np.pi, 0, np.pi)

    # Apply radius constraint if specified
    if radius is not None:
        mask = R_squared <= radius**2
        phase_mask = torch.where(mask, phase_mask, torch.zeros_like(phase_mask))

    # Convert to complex phasor
    phase_mask = torch.exp(1j * phase_mask)

    return phase_mask, phase

def create_binary_fresnel_lens_new(grid_size, feature_size, wavelength, focal_length, radius=None):
    """
    Creates a binary Fresnel lens phase mask with an optional radius.

    Parameters:
    - grid_size: tuple (num_y, num_x) representing the grid size.
    - feature_size: tuple (dy, dx) representing the pixel size in meters.
    - wavelength: Wavelength of the light in meters.
    - focal_length: Focal length of the lens in meters.
    - radius: Optional radius of the lens in meters. If None, the full grid is used.

    Returns:
    - phase_mask: Tensor of the phase mask with values 0 or π.
    """
    num_y, num_x = grid_size
    dy, dx = feature_size

    # Create coordinate grids
    y = torch.arange(-num_y / 2, num_y / 2) * dy
    x = torch.arange(-num_x / 2, num_x / 2) * dx
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Radial coordinate
    R_squared = X**2 + Y**2

    # Calculate the phase profile
    phase = -1*(R_squared)/(2*wavelength*focal_length)

    # Binary phase mask: 0 or π
    phase_mask = torch.where(torch.remainder(phase, 2 * np.pi) < np.pi, 0, np.pi)

    # Apply radius constraint if specified
    if radius is not None:
        mask = R_squared <= radius**2
        phase_mask = torch.where(mask, phase_mask, torch.zeros_like(phase_mask))

    # Convert to complex phasor
    phase_mask = torch.exp(1j * phase_mask)

    return phase_mask, phase


class FT_Lens(nn.Module):
    def __init__(self,delta_x_in, XY_input_grid, wavelength):
        super().__init__()
        self.focal_length = nn.Parameter(torch.tensor(20* mm) )
        self.focal_length.requires_grad = False
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
        input_field = input_field
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

        # output_field = ElectricField(torch.abs(out), torch.angle(out), self.XY_output_grid)

        return out
    

import torch
import torch.nn as nn
import numpy as np

class ASMPropagation(nn.Module):
    def __init__(self, feature_size, wavelength, z, img_shape):
        """
        Initializes the ASMPropagation class.

        Parameters:
        - feature_size: tuple (dy, dx) representing the size of each pixel in meters.
        - wavelength: Wavelength of the light in meters.
        - z: Propagation distance in meters.
        - img_shape: tuple (num_y, num_x) representing the shape of the input field.
        """
        super().__init__()
        self.feature_size = feature_size  # (dy, dx)
        self.wavelength = wavelength
        self.z = z
        self.img_shape = img_shape  # (num_y, num_x)

        # Number of pixels
        num_y, num_x = img_shape
        dx = feature_size
        dy = feature_size

        # Frequencies in cycles per unit length
        fy = torch.fft.fftfreq(num_y, d=dy)  # shape (num_y,)
        fx = torch.fft.fftfreq(num_x, d=dx)  # shape (num_x,)

        # Create the grid
        FY, FX = torch.meshgrid(fy, fx, indexing='ij')  # FY and FX have shapes (num_y, num_x)

        # Compute the squared spatial frequencies
        FX2 = FX**2
        FY2 = FY**2

        # Wavenumber
        k = 2 * np.pi / wavelength

        # Compute the square root argument
        sqrt_arg = 1 - (wavelength**2) * (FX2 + FY2)

        # Avoid evanescent waves by setting negative values to zero
        sqrt_arg = torch.clamp(sqrt_arg, min=0)

        # Compute the transfer function H
        sqrt_term = torch.sqrt(sqrt_arg).to("cpu")
        H = torch.exp(1j * k * z * sqrt_term)

        # Register H as a buffer so it moves with the model (e.g., to GPU if needed)
        self.register_buffer('H', H)

    def forward(self, u_in):
        """
        Propagates the input field using the Angular Spectrum Method.

        Parameters:
        - u_in: Complex tensor of the input field with shape (..., num_y, num_x).

        Returns:
        - u_out: Complex tensor of the propagated field with the same shape as u_in.
        """
        # Compute the Fourier transform of the input field
        U1 = torch.fft.fft2(u_in)

        H = self.H.to(u_in.device)

        # Multiply by the transfer function H
        U2 = U1 * H

        # Compute the inverse Fourier transform to get the output field
        u_out = torch.fft.ifft2(U2)

        return u_out
    
def propagate_and_collect(u_in_masked, z_values):
    intensity_patterns = []
    for z in z_values:
        # Initialize the ASMPropagation class with the current z
        asm = ASMPropagation(feature_size, wavelength, z, grid_size)
        
        # Propagate the field
        u_out = asm(u_in_masked)
        
        # Compute the intensity pattern (magnitude squared)
        intensity = torch.abs(u_out)**2
        
        # Normalize the intensity for visualization
        intensity /= intensity.max()
        
        intensity_patterns.append(intensity.cpu().numpy())
    return intensity_patterns


def propagate_and_collect_side_view(u_in_masked,delta_x_in, wavelength, z_values,grid_size):

    intensity_patterns = []
    for z in z_values:
        asm = ASMPropagation(delta_x_in, wavelength, z, grid_size)
        u_out = asm(u_in_masked)
        intensity = torch.abs(u_out)**2

        # intensity /= intensity.max()
        
        # Take the central slice along the y-axis
        central_slice = intensity[:, intensity.shape[1]//2].detach().cpu().numpy()
        intensity_patterns.append(central_slice)
    
    # Stack the slices to create a 2D array
    return np.stack(intensity_patterns)

def create_gaussian_beam(grid_size, feature_size, wavelength, waist):
    """
    Creates a Gaussian beam with a specified waist.

    Parameters:
    - grid_size: tuple (num_y, num_x) representing the grid size.
    - feature_size: tuple (dy, dx) representing the pixel size in meters.
    - wavelength: Wavelength of the light in meters.
    - waist: Beam waist (w0) in meters.

    Returns:
    - u_in: Complex tensor representing the Gaussian beam.
    """
    num_y, num_x = grid_size
    dy, dx = feature_size

    # Create coordinate grids
    y = torch.arange(-num_y / 2, num_y / 2) * dy
    x = torch.arange(-num_x / 2, num_x / 2) * dx
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Radial coordinate
    R_squared = X**2 + Y**2

    # Gaussian beam amplitude
    amplitude = torch.exp(-R_squared / waist**2)

    # Convert to complex field
    u_in = amplitude.to(torch.cfloat)

    return u_in

if __name__ == "__main__":
    # Simulation parameters
    wavelength = 850e-9  # 633 nm (He-Ne laser)
    feature_size = (2e-6, 2e-6)  # Pixel size (dy, dx) in meters
    grid_size = (512, 512)  # Grid size (num_y, num_x)
    focal_length = 20*mm  # Focal length in meters
    beam_waist = 1000*um  # Beam waist in meters

    # Create the Gaussian beam input field
    u_in = create_gaussian_beam(grid_size, feature_size, wavelength, beam_waist)

    print(grid_size, feature_size, wavelength, focal_length, 500*um)
    # Generate the binary Fresnel lens phase mask
    phase_mask, _ = create_binary_fresnel_lens(grid_size, feature_size, wavelength, focal_length, radius=500*um)
    print(phase_mask)
    # Plot the phase mask
    plt.figure(figsize=(8, 6))
    plt.imshow(phase_mask.angle().cpu().numpy(), cmap='viridis')
    plt.title('Binary Fresnel Lens Phase Mask')
    plt.colorbar(label='Phase (radians)')
    plt.show()

    # Plot the input Gaussian beam intensity
    plt.figure(figsize=(8, 6))
    plt.imshow(torch.abs(u_in)**2, cmap='viridis')
    plt.title(f'Input Gaussian Beam Intensity (w0 = {beam_waist*1e6:.1f} µm)')
    plt.colorbar(label='Intensity (a.u.)')
    plt.show()

    print(u_in)
    # Apply the phase mask to the input field
    u_in_masked = u_in * phase_mask

    # Modify z_values to have more points for a smoother visualization
    z_values = np.linspace(0*mm, 24*mm, 200)

    # Start timing
    start_time = time.time()

    # Generate the side view intensity pattern

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(torch.abs(u_in_masked)**2, cmap='viridis')
    ax[0].set_title('Input Gaussian Beam Intensity (w0 = {beam_waist*1e6:.1f} µm)')
    ax[1].imshow(phase_mask.angle().cpu().numpy(), cmap='viridis')
    ax[1].set_title('Binary Fresnel Lens Phase Mask')
    plt.show()

    print(feature_size[0])
    print(wavelength)
    print(grid_size)

    side_view_intensity = propagate_and_collect_side_view(u_in_masked, feature_size[0], wavelength, z_values,grid_size)

    # End timing
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Propagation simulation took {elapsed_time:.2f} seconds")

    # Plot the side view
    fig, ax = plt.subplots(figsize=(12, 6))
    extent = [z_values[0], z_values[-1], -grid_size[0]//2, grid_size[0]//2]
    im = ax.imshow(side_view_intensity.T, aspect='auto', extent=extent, origin='lower', cmap='hot')
    ax.set_title('Side View of Gaussian Beam Propagation')
    ax.set_xlabel('z distance (m)')
    ax.set_ylabel('y position (pixels)')
    fig.colorbar(im, ax=ax, label='Normalized Intensity')
    plt.show()

    # # Propagate and collect intensity patterns
    # intensity_patterns = propagate_and_collect(u_in_masked, z_values)

    # # Plot the results
    # fig, axs = plt.subplots(1, len(z_values), figsize=(15, 5))
    # for i, (z, intensity) in enumerate(zip(z_values, intensity_patterns)):
    #     ax = axs[i]
    #     im = ax.imshow(intensity, cmap='hot', extent=(-grid_size[1]/2, grid_size[1]/2, -grid_size[0]/2, grid_size[0]/2))
    #     ax.set_title(f'z = {z} m')
    #     ax.axis('off')
    # fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
    # plt.show()