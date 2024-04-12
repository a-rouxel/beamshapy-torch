import torch
from torch import nn
from units import *
from electric_field import ElectricField

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