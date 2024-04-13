import torch


def generate_profile(profile_type, XY_grid,radius=None, parabola_coef=None,angle=0,width=None, height=None, position=(0, 0), period=None, phase_offset=0):

    if profile_type == "Fresnel Lens":
        target_intensity = fresnel_lens(XY_grid, radius, parabola_coef)
        return target_intensity
    elif profile_type == "Rectangle":
        target_intensity = RectangularMask(XY_grid, angle, width, height, position)
        return target_intensity
    elif profile_type == "Sinus":
        target_intensity = SinusMask(XY_grid, period, angle, phase_offset)
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

def SinusMask(XY_grid, period, angle, phase_offset=0):
    """
    Generates a sinusoidal amplitude array using PyTorch.

    Args:
        XY_grid (torch.Tensor): Tensor of shape [2, height, width] containing X and Y coordinates
        period (float): Period of the sinusoidal amplitude array (in m).
        angle (float): Rotation angle of the sinusoidal amplitude array (in rad).
        phase_offset (float): Phase offset of the sinusoidal amplitude array (in rad).

    Returns:
        torch.Tensor: 2D tensor with the mask.
    """
    angle = torch.tensor(angle)
    phase_offset = torch.tensor(phase_offset)

    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    rotated_X = XY_grid[0] * cos_angle - XY_grid[1] * sin_angle

    mask = torch.sin(2 * torch.pi * rotated_X / period + phase_offset)
    return mask

