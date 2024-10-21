import torch
from helpers import load_yaml_config
from units import *
from electric_field import ElectricField
import matplotlib.pyplot as plt

def generate_profile(profile_type, XY_grid,radius=None, parabola_coef=None,angle=0,width=None, height=None, position=(0, 0), period=None, phase_offset=0):

    if profile_type == "Fresnel Lens":
        target_intensity = fresnel_lens(XY_grid, radius, parabola_coef)
        return target_intensity
    elif profile_type == "Rectangle":
        target_intensity = RectangularMask(XY_grid, angle, width, height, position)
        return target_intensity
    elif profile_type == "Sinus":
        target_intensity = SinusMask(XY_grid, period, angle, phase_offset, position)
        return target_intensity
    

def fresnel_lens(XY_grid, radius, parabola_coef):
    """
    Function to generate a Fresnel lens intensity profile using PyTorch

    Args:
        XY_grid (torch.Tensor): Tensor of shape [2, height, width] containing X and Y coordinates of the target grid
        radius (float): Radius of the Fresnel lens (in m)
        parabola_coef (float): Coefficient of the parabola profile (no units)

    Returns:
        torch.Tensor: Fresnel lens intensity profile
    """

    GridPositionMatrix_X_out, GridPositionMatrix_Y_out = XY_grid[0], XY_grid[1]
    parabola = ParabolaMask(GridPositionMatrix_X_out, GridPositionMatrix_Y_out, parabola_coef)
    wrap_parabola = torch.remainder(parabola, 1)

    mask = GridPositionMatrix_X_out**2 + GridPositionMatrix_Y_out**2 <= radius**2
    wrap_parabola = torch.where(mask, wrap_parabola, torch.zeros_like(wrap_parabola))

    return wrap_parabola

def ParabolaMask(GridPositionMatrix_X_out, GridPositionMatrix_Y_out, coef):
    """
    Generates a parabolic phase mask using PyTorch.

    Args:
        GridPositionMatrix_X_out (torch.Tensor): 2D tensor with the x coordinates of the grid (in m).
        GridPositionMatrix_Y_out (torch.Tensor): 2D tensor with the y coordinates of the grid (in m).
        coef (float): Coefficient for the parabolic equation (in m^-1).

    Returns:
        torch.Tensor: 2D tensor with the mask.
    """

    Z_out = coef * (GridPositionMatrix_X_out ** 2 + GridPositionMatrix_Y_out ** 2)

    return Z_out


def RectangularMask(XY_grid, angle, width, height, position=(0, 0)):
    """
    Generates a rectangular amplitude mask using PyTorch.

    Args:
        XY_grid (torch.Tensor): Tensor of shape [2, height, width] containing X and Y coordinates
        angle (float): Rotation angle of the rectangle (in rad).
        width (float): Width of the rectangle (in m).
        height (float): Height of the rectangle (in m).
        position (tuple): Center position of the rectangle (in m).

    Returns:
        torch.Tensor: 2D tensor with the mask.
    """

    angle = torch.tensor(angle)
    width = torch.tensor(width)
    height = torch.tensor(height)

    x0, y0 = position
    shifted_X = XY_grid[0] - x0
    shifted_Y = XY_grid[1] - y0

    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    rotated_X = shifted_X * cos_angle - shifted_Y * sin_angle
    rotated_Y = shifted_Y * cos_angle + shifted_X * sin_angle

    mask = torch.zeros_like(rotated_X)
    mask[(torch.abs(rotated_X) < width / 2) & (torch.abs(rotated_Y) < height / 2)] = 1

    return mask

def SinusMask(XY_grid, period, angle, phase_offset=0, initial_position=(0, 0)):
    """
    Generates a sinusoidal amplitude array using PyTorch.

    Args:
        XY_grid (torch.Tensor): Tensor of shape [2, height, width] containing X and Y coordinates
        period (float): Period of the sinusoidal amplitude array (in m).
        angle (float): Rotation angle of the sinusoidal amplitude array (in rad).
        phase_offset (float): Phase offset of the sinusoidal amplitude array (in rad).
        initial_position (tuple): Initial position to shift the grid coordinates (in m).

    Returns:
        torch.Tensor: 2D tensor with the mask.
    """
    angle = torch.tensor(angle)
    phase_offset = torch.tensor(phase_offset)

    # Adjust the grid coordinates by the initial position
    shifted_X = XY_grid[0] - initial_position[0]
    shifted_Y = XY_grid[1] - initial_position[1]

    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    rotated_X = shifted_X * cos_angle - shifted_Y * sin_angle

    mask = torch.sin(2 * torch.pi * rotated_X / period + phase_offset)
    return mask


def generate_target_profiles(yaml_file,XY_grid,list_modes_nb=[]):

    config = load_yaml_config(yaml_file)

    width = config["width"]*um
    height = config["height"]*um


    list_target_profiles = []
    for mode_nb in list_modes_nb:
        if mode_nb %2 == 0:
            phase_offset = torch.pi/2
        else:
            phase_offset = 0
        sinus_period = 2* width / (1+mode_nb)

        target_field = generate_profile("Rectangle",XY_grid,width=width,height=height)
        target_field *= generate_profile("Sinus",XY_grid,period=sinus_period,phase_offset=phase_offset)

        target_field = ElectricField(torch.abs(target_field), torch.angle(target_field), XY_grid)
        list_target_profiles.append(target_field.field)

    return list_target_profiles


def generate_target_profiles_specific_modes(yaml_file,XY_grid,list_modes_nb=[],orientation="vertical"):


    # width = config["width"]*um
    list_width = [61.5*um,61.5*um,61.5*um,61.5*um,60.5*um,60.0*um,59.0*um,58.5*um,57.5*um,57.0*um]
    height = 8*um


    list_target_profiles = []
    for mode_nb in list_modes_nb:
        width = list_width[mode_nb]
        if mode_nb %2 == 0:
            phase_offset = torch.pi/2
        else:
            phase_offset = 0
        sinus_period = 2* width / (1+mode_nb)

        target_field = generate_profile("Rectangle",XY_grid,width=width,height=height)
        target_field *= generate_profile("Sinus",XY_grid,period=sinus_period,phase_offset=phase_offset)

        if orientation == "vertical":
            pass
        elif orientation == "horizontal":
            target_field = target_field.T

        target_field = ElectricField(torch.abs(target_field), torch.angle(target_field), XY_grid)
        list_target_profiles.append(target_field.field)

    return list_target_profiles


def generate_target_profiles_specific_modes_2D(yaml_file,XY_grid,list_modes_nb=[]):

    config = load_yaml_config(yaml_file)

    # width = config["width"]*um
    list_width = [61.5*um,61.5*um,61.5*um,61.5*um,60.5*um,60.0*um,59.0*um,58.5*um,57.5*um,57.0*um]
    list_height = [61.5*um,61.5*um,61.5*um,61.5*um,60.5*um,60.0*um,59.0*um,58.5*um,57.5*um,57.0*um]


    list_target_profiles = []

    for mode_nb_x, mode_nb_y in list_modes_nb:
        width = list_width[mode_nb_x]
        height = list_height[mode_nb_y]

        if mode_nb_x %2 == 0:
            phase_offset_x = torch.pi/2
        else:
            phase_offset_x = 0
        if mode_nb_y %2 == 0:
            phase_offset_y = torch.pi/2
        else:
            phase_offset_y = 0
        sinus_period_x = 2* width / (1+mode_nb_x)
        sinus_period_y = 2* height / (1+mode_nb_y)

        target_field = generate_profile("Rectangle",XY_grid,width=width,height=height)
        target_field *= generate_profile("Sinus",XY_grid,period=sinus_period_x,phase_offset=phase_offset_x)
        target_field *= generate_profile("Sinus",XY_grid,period=sinus_period_y,phase_offset=phase_offset_y)

        target_field = ElectricField(torch.abs(target_field), torch.angle(target_field), XY_grid)
        list_target_profiles.append(target_field.field)

        plt.imshow(torch.abs(target_field).detach().numpy())
        plt.show()

    return list_target_profiles



def generate_target_profiles_shifted(yaml_file,XY_grid,position_shift,list_modes_nb=[]):

    config = load_yaml_config(yaml_file)

    width = config["width"]*um
    height = config["height"]*um


    list_target_profiles = []
    for mode_nb in list_modes_nb:
        if mode_nb %2 == 0:
            phase_offset = torch.pi/2
        else:
            phase_offset = 0
        sinus_period = 2* width / (1+mode_nb)

        target_field = generate_profile("Rectangle",XY_grid,width=width,height=height,position=position_shift)
        target_field *= generate_profile("Sinus",XY_grid,period=sinus_period,phase_offset=phase_offset,position=position_shift)

        target_field = ElectricField(torch.abs(target_field), torch.angle(target_field), XY_grid)
        list_target_profiles.append(target_field.field)

    return list_target_profiles


def generate_target_profile_CRIGF(list_mode_nb=(2,2),XY_grid=None):



    list_width = [61.5*um,61.5*um,61.5*um,61.5*um,60.5*um,60.0*um,59.0*um,58.5*um,57.5*um,57.0*um]
    list_height = [61.5*um,61.5*um,61.5*um,61.5*um,60.5*um,60.0*um,59.0*um,58.5*um,57.5*um,57.0*um]


    mode_nb_x = list_mode_nb[0]
    mode_nb_y = list_mode_nb[1]


    list_target_profiles = []

    width = list_width[mode_nb_x]
    height = list_height[mode_nb_y]



    target_field = generate_profile("Rectangle",XY_grid,width=width,height=height)

    target_field = ElectricField(torch.abs(target_field), torch.angle(target_field), XY_grid)
    list_target_profiles.append(target_field.field)

    return list_target_profiles



