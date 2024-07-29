import os
import yaml
from functions_basic_shapes import *
import torch
from scipy.optimize import brentq

nm = 1e-9

def wrap_phase(phase):
    """
    Wraps the phase between -pi and pi.

    Args:
        phase (np.ndarray): 2D array with the phase (in rad).

    Returns:
        np.ndarray: 2D array with the wrapped phase (in rad).

    """

    return np.angle(np.exp(1j * phase))

def theorical_deformation_sinc(x):
    """
    Computes the theorical deformation of the sinc function.

    Args:
        x (np.ndarray): 1D array with the x values.

    Returns:
        np.ndarray: 1D array with the theorical deformation of the sinc function.

    """

    a = np.sin(np.pi * (1 - x))
    b = (np.pi * (1 - x))
    theorical_amplitude_modulation = np.divide(a, b, out=np.ones_like(a), where=b != 0)

    return theorical_amplitude_modulation

def root_theorical_deformation_sinc(x,c):
    """
    Computes the root of the theorical deformation of the sinc function.

    Args:
        x (np.ndarray): 1D array with the x values.
        c (float): Value of the theorical deformation of the sinc function.

    """

    return theorical_deformation_sinc(x) - c


def load_yaml_config(file_path):
    """
    Load a YAML configuration file.

    Args:
        file_path (str): Path to the YAML file

    Returns
        dict: A dictionary containing the configuration data
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def generate_target_amplitude(Grid_XY_in, Grid_XY_out, wavelength,focal_length,
                              amplitude_type, period=0, position=(0, 0), scale_factor=1,
                              angle=0, width=0, height=0, coef=None, sigma=0, n=0,
                              amplitude_path=None, phase_offset=0):

    """
    Main function for generating target amplitude profiles

    Args:
        amplitude_type (str): Type of amplitude profile to generate
        period (float): Period of the amplitude profile (in m)
        position (tuple): Position of the amplitude profile (in m)
        scale_factor (float): Scale factor of the amplitude profile
        angle (float): Rotation angle of the amplitude profile (in rad)
        width (float): Width of the rectange profile (in m)
        height (float): Height of the rectange profile (in m)
        coef (float): Coefficient of the parabola profile (in no units)
        sigma (float): Sigma of the Gaussian profile (in m)
        n (int): Order of the supergaussian profile
        amplitude_path (str): Path to the amplitude profile H5 file
        phase_offset (float): Phase offset of the sinusoidal profile (in rad)
    """


    if amplitude_type == "Rectangle":
        amplitude = RectangularMask(Grid_XY_out[0],Grid_XY_out[1], angle,
                                    width, height, position=position)

    elif amplitude_type == "Gaussian":
        amplitude = supergaussian2D(Grid_XY_in[1][0,:], n, sigma)


    elif amplitude_type == "Wedge":
        x_proj = np.cos(angle) * position
        y_proj = np.sin(angle) * position

        amplitude_x = Simple2DWedgeMask(Grid_XY_in[1][0,:], wavelength, x_proj,
                                        focal_length)
        amplitude_y = np.flip(
            np.transpose(Simple2DWedgeMask(Grid_XY_in[1][0,:], wavelength, y_proj,
                                           focal_length)), 0)
        amplitude = amplitude_x + amplitude_y


    elif amplitude_type == "Parabola":
        amplitude = ParabolaMask(Grid_XY_out[0],Grid_XY_out[1], coef)

    elif amplitude_type == "Sinus":
        amplitude = SinusMask(Grid_XY_out[0], Grid_XY_out[1], period, angle, phase_offset)

    elif amplitude_type == "Cosinus":
        amplitude = CosinusMask(Grid_XY_out[0], Grid_XY_out[1], period,  angle)

    else:
        print("amplitude type not recognized")
        amplitude = None

    return torch.tensor(amplitude)

import matplotlib.pyplot as plt
def generate_target_mask(inverse_fourier_target_field,mask_type,input_field=None,threshold=0.001, correction_a_values=None,correction_tab=None,amplitude_factor=1):
    """
    Generate a mask based on a Target Amplitude or Intensity.

    Args:
        mask_type (str): Type of mask to generate. Available options are:
            - "ϕ target field": Phase mask based on the target field
            - "modulation amplitude": Amplitude mask based on the target field
        amplitude_factor (float): Factor to multiply the target amplitude by. Default is 1.

    Returns:
        np.array: Mask to be applied to the SLM

    """

    if mask_type == "phase target field":
        target_field = inverse_fourier_target_field
        mask = np.angle(target_field)
        mask[mask < -np.pi+1e-3] = np.pi
        mask[np.abs(mask) < 1e-3] = 0
        return mask

    if mask_type == "modulation amplitude":
        # power_inverse_fourier_target_field = torch.sum(torch.sum(torch.abs(inverse_fourier_target_field)**2))
        # power_input_field = torch.sum(torch.sum(torch.abs(input_field)**2))
        #
        # print(power_inverse_fourier_target_field,power_input_field)
        max_input_field = torch.max(torch.abs(input_field))
        max_fourier_target_field = torch.max(torch.abs(inverse_fourier_target_field))

        normalized_target_field = inverse_fourier_target_field *max_input_field / max_fourier_target_field

        target_abs_amplitude = torch.abs(normalized_target_field)
        input_abs_amplitude = torch.abs(input_field)



        mask = WeightsMask(input_abs_amplitude,target_abs_amplitude,threshold)


        a_values, correction_tab = generate_correction_tab()
        corrected_mask = correct_modulation_values(mask, a_values, correction_tab)

        # plt.plot(target_abs_amplitude[:, 1250].detach().numpy())
        # plt.plot(input_abs_amplitude[:, 1250].detach().numpy())
        # plt.plot(mask[:, 1250].detach().numpy())
        # plt.plot(corrected_mask[:, 1250])
        # plt.show()

        return corrected_mask,mask,normalized_target_field

    else:
        print("mask_type not recognized")
        return None


def design_mask(Grid_XY_in,mask_type,wavelength,focal_length,period=None,position = None, charge=None,orientation=None,angle = None, width = None, height = None, sigma_x=None,sigma_y=None,threshold=None,mask_path=None,amplitude_factor=1):

    """
    Design a mask based on the parameters provided.

    Avalailable mask types are:
        - Grating
        - Gaussian
        - Vortex
        - Wedge
        - Custom h5 Mask
        - Rectangle
        - Phase reversal
        - Phase Jump

    Args:
        mask_type (str): Type of mask to generate
        period (float): Period of the mask (in m)
        position (float): Position of the mask (in m)
        charge (int): Charge of the vortex mask
        orientation (str): Orientation of the mask. Can be "Horizontal" or "Vertical"
        angle (float): Angle of the mask (in rad)
        width (float): Width of the rectangle (in m)
        height (float): Height of the rectangle (in m)
        sigma_x (float): Sigma of the Gaussian mask in the x direction (in m)
        sigma_y (float): Sigma of the Gaussian mask in the y direction (in m)
        threshold (float): Threshold of the mask
        mask_path (str): Path to the custom mask
        amplitude_factor (float): Amplitude factor of the mask

    Returns:
        np.array: Generated mask
    """

    if Grid_XY_in[1][0,:] is None:
        raise ValueError("Please generate Input Beam first")

    if mask_type == "Grating":
        M1 = Simple1DBlazedGratingMask(Grid_XY_in[1][0,:], period)
        mask = np.tile(M1, (Grid_XY_in[1][0,:].shape[0], 1))
        if orientation == "Vertical":
            mask = np.transpose(mask)
        return mask

    if mask_type == "Gaussian":

        sigma_x *= 10**-6
        sigma_y *= 10**-6

        if sigma_x is None or sigma_y is None:
            raise ValueError("Please provide values for sigma_x and sigma_y for the Gaussian mask.")

        x, y = np.meshgrid(Grid_XY_in[1][0,:], Grid_XY_in[1][0,:])
        mask = np.exp(-((x) ** 2 / (2 * sigma_x ** 2) + (y) ** 2 / (2 * sigma_y ** 2)))

        return mask

    if mask_type == "Vortex":
        mask = VortexMask(Grid_XY_in[1][0,:], charge)

        return mask

    if mask_type == "Wedge":
        x_proj = np.cos(angle)*position
        y_proj = np.sin(angle)*position

        mask_x = Simple2DWedgeMask(Grid_XY_in[1][0,:],wavelength,x_proj,focal_length)
        mask_y = np.flip(np.transpose(Simple2DWedgeMask(Grid_XY_in[1][0,:],wavelength,y_proj,focal_length)),0)
        mask = mask_x + mask_y

        return mask



    if mask_type == "Rectangle":
        mask = RectangularMask(Grid_XY_in[0],Grid_XY_in[1],angle, width,height)
        return mask

    if mask_type == "Phase Jump":
        mask = PiPhaseJumpMask(Grid_XY_in[0],Grid_XY_in[1],orientation, position)
        return mask

    if mask_type == "Phase Reversal":
        sigma_x = sigma_x
        sigma_y = sigma_y
        mask = PhaseReversalMask(Grid_XY_in[0],Grid_XY_in[1],2*mm,sigma_x,sigma_y)
        # self.phase_inversed_Field = SubPhase(self.beam_shaper.input_beam,mask)

        return mask


    # if mask_type == "Custom h5 Mask":
    #     if mask_path is None:
    #         raise ValueError("Please provide h5 file path for custom mask.")
    #
    #     with h5py.File(mask_path, 'r') as f:
    #         mask = f['mask'][:]
    #
    #     # If the mask is too small, center it in a new array matching the GridPositionMatrix dimensions
    #     # If the mask is too small, center it in a new array matching the GridPositionMatrix dimensions
    #     if mask.shape != Grid_XY_in[0].shape:
    #         new_mask = np.zeros_like(Grid_XY_in[0])
    #         x_offset = (new_mask.shape[0] - mask.shape[0]) // 2
    #         y_offset = (new_mask.shape[1] - mask.shape[1]) // 2
    #         new_mask[x_offset: x_offset + mask.shape[0], y_offset: y_offset + mask.shape[1]] = mask
    #         mask = new_mask
    #
    #     else:
    #         print("mask_type not recognized")
    #     return mask


def generate_correction_tab(nb_of_samples=1000,func=root_theorical_deformation_sinc):
    """Generates the correction tab corresponding to the sinc. Cf. Modulation amplitude of a blazed grating.pdf

    Args:
        nb_of_samples (int, optional): Number of samples for the correction tab. Defaults to 1000.
        func (function, optional): Function to use for the correction tab. Defaults to root_theorical_deformation_sinc.

    Returns:
        a_values (np.array): Array of values between 0 and 1 for the correction tab.
        correction_tab (np.array): Array of correction values for the correction tab.

    """

    a_values = np.linspace(0.001, 0.999, nb_of_samples - 2)
    correction_tab = np.zeros_like(a_values)

    for i, a in enumerate(a_values):
        correction_tab[i] = brentq(func, 0, 1, args=(a,))

    a_values = list(a_values)
    a_values.insert(0, 0)
    a_values.append(1)
    correction_tab = list(correction_tab)
    correction_tab.insert(0, 0)
    correction_tab.append(1)

    return a_values, correction_tab

def theorical_deformation_sinc(x):
    """
    Computes the theorical deformation of the sinc function.

    Args:
        x (np.ndarray): 1D array with the x values.

    Returns:
        np.ndarray: 1D array with the theorical deformation of the sinc function.

    """

    a = np.sin(np.pi * (1 - x))
    b = (np.pi * (1 - x))
    theorical_amplitude_modulation = np.divide(a, b, out=np.ones_like(a), where=b != 0)

    return theorical_amplitude_modulation

def root_theorical_deformation_sinc(x,c):
    """
    Computes the root of the theorical deformation of the sinc function.

    Args:
        x (np.ndarray): 1D array with the x values.
        c (float): Value of the theorical deformation of the sinc function.

    """

    return theorical_deformation_sinc(x) - c

# def generate_correction_tab(step,func):
#     """
#     Generates the correction table.
#
#     Args:
#         step (int): Number of points of the correction table.
#         func (function): Function to correct.
#
#     Returns:
#         np.ndarray: 1D array with the correction a values (between 0 and 1).
#
#     """
#
#     a_values = np.linspace(0.001,0.999,step-2)
#     correction_tab = np.zeros_like(a_values)
#
#     for i,a in enumerate(a_values):
#         correction_tab[i] = brentq(func, 0, 1, args=(a,))
#
#     a_values = list(a_values)
#     a_values.insert(0,0)
#     a_values.append(1)
#     correction_tab = list(correction_tab)
#     correction_tab.insert(0,0)
#     correction_tab.append(1)
#
#     return a_values,correction_tab

def correct_modulation_values(modulation_values,a_values,correction_tab):
    """
    Interpolates the correction table

    Args:
        modulation_values (np.ndarray): 2D array with the modulation values.
        a_values (np.ndarray): 1D array with the correction a values (between 0 and 1).
        correction_tab (np.ndarray): 2D array with the correction table.

    Returns:
        np.ndarray: 2D array with the corrected modulation values.

    """

    return np.interp(modulation_values,a_values,correction_tab)


def WeightsMask(input_amplitude, target_amplitude, threshold=10 ** -1):
    # Ensure input_amplitude and target_amplitude are tensors
    input_amplitude = torch.tensor(input_amplitude, dtype=torch.float32)
    target_amplitude = torch.tensor(target_amplitude, dtype=torch.float32)

    # Create a mask for where input_amplitude is greater than the threshold
    mask = torch.abs(input_amplitude) > threshold

    # Use the mask to safely divide target_amplitude by input_amplitude
    weights = torch.ones_like(target_amplitude)
    weights[mask] = torch.abs(target_amplitude[mask] / input_amplitude[mask])

    # Clip weights to be no greater than 1
    weights = torch.clamp(weights, max=1)

    return weights