import cProfile
import os
import pstats
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
from pprint import pprint
from helpers import load_yaml_config
from simulation import Simulation
from sources import Source
from slm import SLM
from ft_lens import FT_Lens, ASMPropagation
from units import *
from FieldGeneration import generate_target_profiles
from cost_functions import calculate_normalized_overlap, quantize_phase
from helpers import generate_target_amplitude, design_mask, generate_target_mask, wrap_phase

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
plt.rcParams['text.usetex'] = True

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

def format_val(val):
    """Format the value to scientific notation if smaller than 0.005, else to two decimal places."""
    return f'{val:.1e}' if val < 0.005 else f'{val:.2f}'

def compute_loss_no_weights(out_field, list_target_field):

    list_overlaps = []

    for idx, target_field in enumerate(list_target_field):

        overlap = calculate_normalized_overlap(out_field, target_field)


        list_overlaps.append(float(overlap))


    return list_overlaps


def floyd_steinberg(image):
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[y, x]
            new = np.round(old)
            image[y, x] = new
            error = old - new
            if x + 1 < w:
                image[y, x + 1] += error * 0.4375  # right, 7 / 16
            if (y + 1 < h) and (x + 1 < w):
                image[y + 1, x + 1] += error * 0.0625  # right, down, 1 / 16
            if y + 1 < h:
                image[y + 1, x] += error * 0.3125  # down, 5 / 16
            if (x - 1 >= 0) and (y + 1 < h):
                image[y + 1, x - 1] += error * 0.1875  # left, down, 3 / 16
    return image



def main():

    from matplotlib.colors import LinearSegmentedColormap, \
        LogNorm  # Create a new colormap from the hot colormap without the white color
    hot_cmap = plt.cm.get_cmap('hot')
    non_linear_points = np.linspace(0, 0.9, 256) ** 2  # Quadratic distribution
    new_cmap = LinearSegmentedColormap.from_list('trunc_hot', hot_cmap(non_linear_points))

    dir_path = "./comparisons_results_optim_on_selectivity_3um_with_optim/"
    os.makedirs(dir_path, exist_ok=True)

    nb_of_modes_to_consider = 9

    cross_correlation_matrix_davis = np.zeros((nb_of_modes_to_consider, nb_of_modes_to_consider))
    cross_correlation_matrix_phase_inversion = np.zeros((nb_of_modes_to_consider, nb_of_modes_to_consider))
    cross_correlation_matrix_floyd_steinberg = np.zeros((nb_of_modes_to_consider, nb_of_modes_to_consider))
    cross_correlation_matrix_optim = np.zeros((nb_of_modes_to_consider, nb_of_modes_to_consider))
    cross_correlation_matrix_asm = np.zeros((nb_of_modes_to_consider, nb_of_modes_to_consider))
    cross_correlation_matrix_phase_inversion_only = np.zeros((nb_of_modes_to_consider, nb_of_modes_to_consider))
    
    list_cross_correlation_matrix = [cross_correlation_matrix_davis, 
                                     cross_correlation_matrix_phase_inversion_only, 
                                     cross_correlation_matrix_phase_inversion, 
                                     cross_correlation_matrix_floyd_steinberg, 
                                     cross_correlation_matrix_asm,
                                     cross_correlation_matrix_optim]

    config_source = load_yaml_config("./configs/source.yml")
    config_target = load_yaml_config("./configs/target_profile.yml")
    config_slm = load_yaml_config("./configs/SLM.yml")
    config_simulation = load_yaml_config("./configs/simulation.yml")
    config_input_beam = load_yaml_config("./configs/input_beam.yml")
    config_optical_system = load_yaml_config("./configs/optical_system.yml")


    simulation = Simulation(config_dict=config_simulation)
    source = Source(config_dict=config_source, XY_grid=simulation.XY_grid)
    ft_lens = FT_Lens(simulation.delta_x_in, simulation.XY_grid, source.wavelength)

    simulation_asm = Simulation(config_dict=load_yaml_config("./configs/simulation_ASM.yml"))
    source_asm = Source(config_dict=config_source, XY_grid=simulation_asm.XY_grid)
    asm = ASMPropagation(simulation_asm.delta_x_in, source_asm.wavelength, 20*mm, simulation_asm.XY_grid[0].shape) 
    
    list_target_fields = generate_target_profiles(yaml_file="./configs/target_profile.yml",
                                                  XY_grid=ft_lens.XY_output_grid,
                                                  list_modes_nb=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    
    list_target_fields_asm = generate_target_profiles(yaml_file="./configs/target_profile.yml",
                                                  XY_grid=simulation_asm.XY_grid,
                                                  list_modes_nb=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    for mode_nb in range(0, nb_of_modes_to_consider):

        # Implement the comparison using PyTorch
        # Target Amplitude Parameters
        width = 60 * um
        height = 8 * um

        sinus_period = 2 * width / (mode_nb + 1)
        if mode_nb % 2 == 0:
            phase_off = np.pi / 2
            mode_parity = "even"
        else:
            phase_off = 0
            mode_parity = "odd"

        crop_size_detector = 100 * um
        crop_size_slm = 5 * mm
        position = 0.12 * mm

        # OUTPUT GRID MASK
        mask_with_center = (ft_lens.XY_output_grid[1][0, :] > - crop_size_detector / 2) & (ft_lens.XY_output_grid[1][0, :] < crop_size_detector / 2)  # example bounds

        # OUTPUT GRID MASK
        mask_with_wedge_x = (ft_lens.XY_output_grid[1][0, :] > position - crop_size_detector / 2) & (ft_lens.XY_output_grid[1][0, :] < position +crop_size_detector / 2)  # example bounds
        mask_with_wedge_y = (ft_lens.XY_output_grid[1][0, :] > -crop_size_detector / 2) & ( ft_lens.XY_output_grid[1][0, :] < crop_size_detector / 2)  # example bounds

        # SLM MASK ARRAY CENTER
        mask_center_mask = (simulation.XY_grid[1][0,:] > -crop_size_slm / 2) & (simulation.XY_grid[1][0,:] < crop_size_slm / 2)

        # mask CRIGF
        mask_CRIGF_x = (ft_lens.XY_output_grid[1][0, :] > -height/2) & (ft_lens.XY_output_grid[1][0, :] < height/2)
        mask_CRIGF_y = (ft_lens.XY_output_grid[1][0, :] > -width/2) & (ft_lens.XY_output_grid[1][0, :] < width/2)

        mask_CRIGF_switch_x = (ft_lens.XY_output_grid[1][0, :] > position-height/2) & (ft_lens.XY_output_grid[1][0, :] < position+height/2)
        mask_CRIGF_switch_y = (ft_lens.XY_output_grid[1][0, :] > -width/2) & (ft_lens.XY_output_grid[1][0, :] < width/2)


        target_amplitude = generate_target_amplitude(simulation.XY_grid, ft_lens.XY_output_grid, source.wavelength,
                                                     ft_lens.focal_length,
                                                     amplitude_type="Rectangle", width=width, height=height)
        target_amplitude *= generate_target_amplitude(simulation.XY_grid, ft_lens.XY_output_grid, source.wavelength,
                                                      ft_lens.focal_length,
                                                      amplitude_type="Sinus", period=sinus_period, phase_offset=phase_off)
        
 
        inverse_fourier_transform = ft_lens(target_amplitude, pad=False, flag_ifft=True)



        wedge_mask = design_mask(simulation.XY_grid, "Wedge", source.wavelength, ft_lens.focal_length, angle=0,
                                 position=0.12 * mm)
        phase_inversion_mask = generate_target_mask(inverse_fourier_transform, mask_type="phase target field")
        amplitude_modulation_mask, uncorrected_amplitude_mask, _ = generate_target_mask(inverse_fourier_transform,
                                                                                        mask_type="modulation amplitude",
                                                                                        input_field=source.field.field)



        cut_x_phase_inversion = phase_inversion_mask[1500,:]
        cut_x_wedge = wedge_mask[1500,:]
        cut_x_amplitude = amplitude_modulation_mask[1500,:]
        cut_x_uncorrected_amplitude = uncorrected_amplitude_mask[1500,:]
        cut_x_inverse_fourier_transform = torch.abs(inverse_fourier_transform[1500,:]).numpy()


        mask_1 = wrap_phase(phase_inversion_mask + wedge_mask) * _


        if mode_nb % 2 == 0:
            target_field_unnormalized = torch.real(inverse_fourier_transform)
        else:
            target_field_unnormalized = torch.imag(inverse_fourier_transform)

        new_mask = (target_field_unnormalized - torch.min(target_field_unnormalized)) / (
                    torch.max(target_field_unnormalized) - torch.min(target_field_unnormalized))
        target_field_unnormalized /= (torch.max(target_field_unnormalized) * 2)
        new_mask = target_field_unnormalized + 0.5

        new_mask_np = new_mask.numpy()
        dithered_amplitude = torch.tensor(floyd_steinberg(new_mask_np))
        mask_2 = dithered_amplitude * np.pi
        mask_2 -= np.pi / 2



        # 3. Phase inversion + dithering + Fresnel lens (ASM propagation)
        fresnel_mask, _ = create_binary_fresnel_lens(simulation_asm.XY_grid[0].shape,
                                                     (simulation_asm.delta_x_in, simulation_asm.delta_x_in),
                                                     source_asm.wavelength, 20*mm, radius=None)
        mask_3 = torch.angle(fresnel_mask * torch.exp(1j * mask_2)).numpy()

        # Add a new mask for phase inversion alone
        mask_phase_inversion_only = phase_inversion_mask


        # 5 - Optim. with lens
        list_run = [2,1,1,1,3,1,2,2,3]
        mask_optim = np.load(f"/data/Phase_masks_with_lens_V2/phase_masks/mode_{mode_nb}/run_{list_run[mode_nb]}/slm_phase_epoch_4095.npy")

        mask_optim = torch.tensor(mask_optim.copy(), dtype=torch.float32)
        mask_optim_with_quantization = quantize_phase(mask_optim, 2, mode_parity=mode_parity)



        mask_dirs = [
        "davis_masks",
        "phase_inversion_masks",
        "phase_inversion_dithering_masks",
        "phase_inversion_dithering_fresnel_masks",
        "optim_masks",
        "optim_masks_with_quantization"
    ]
        for subdir in mask_dirs:
            os.makedirs(os.path.join(dir_path, subdir), exist_ok=True)

        # Reorder the masks array
        masks = [mask_1, mask_phase_inversion_only, mask_2, mask_3, mask_optim,mask_optim_with_quantization]
        intensities = []
        cropped_intensities = []
        cropped_masks = []
        cropped_amplitudes = []
        cropped_energy_inside = []

        for idx, mask in enumerate(masks):

            mask_filename = os.path.join(dir_path, mask_dirs[idx], f"full_mask_mode_{mode_nb}.npy")
            np.save(mask_filename, mask)

            if idx != 3:  # Use regular propagation for all except ASM
                slm = SLM(config_dict=config_slm, XY_grid=simulation.XY_grid, initial_phase=mask, device="cpu")
                modulated_field = slm.apply_phase_modulation(source.field.field, mapping=False)
                out_field = ft_lens(modulated_field, pad=False, flag_ifft=False)
            else:
                slm_asm = SLM(config_dict=config_slm, XY_grid=simulation_asm.XY_grid, initial_phase=mask, device='cpu')
                modulated_field = slm_asm.apply_phase_modulation(source_asm.field.field, mapping=False)
                out_field = asm(modulated_field)

            intensity = torch.abs(out_field) ** 2

            if idx !=3:
                power_previous = torch.sum(torch.abs(out_field) ** 2)
                print("power_previous",power_previous)
            if idx == 2:
                power_asm = torch.sum(torch.abs(out_field) ** 2)
                print("power_asm",power_asm)
                intensity *= power_previous/power_asm


        

            intensities.append(intensity)

            # Example cropping, modify as needed
            if idx ==0:
                cropped_intensity = intensity[mask_with_wedge_y][:, mask_with_wedge_x]
                cropped_amplitude = out_field[mask_CRIGF_switch_y][:, mask_CRIGF_switch_x]
                cropped_amplitudes.append(cropped_amplitude)
            
            elif idx == 3:
                mask_with_center_asm = (simulation_asm.XY_grid[1][0, :] > - crop_size_detector / 2) & (simulation_asm.XY_grid[1][0, :] < crop_size_detector / 2)  # example bounds
                mask_CRIGF_asm_x = (simulation_asm.XY_grid[1][0, :] > -height/2) & (simulation_asm.XY_grid[1][0, :] < height/2)
                mask_CRIGF_asm_y = (simulation_asm.XY_grid[1][0, :] > -width/2) & (simulation_asm.XY_grid[1][0, :] < width/2)

                mask_center_asm = (simulation_asm.XY_grid[1][0,:] > -crop_size_slm / 2) & (simulation_asm.XY_grid[1][0,:] < crop_size_slm / 2)

                cropped_intensity = intensity[mask_with_center_asm,:][:, mask_with_center_asm]
                cropped_amplitude = out_field[mask_CRIGF_asm_y][:, mask_CRIGF_asm_x]
                cropped_amplitudes.append(cropped_amplitude)
            else :
                cropped_intensity = intensity[mask_with_center,:][:, mask_with_center]
                cropped_amplitude = out_field[mask_CRIGF_y][:, mask_CRIGF_x]
                cropped_amplitudes.append(cropped_amplitude)


            cropped_energy_inside.append(torch.sum(torch.abs(cropped_amplitude)**2)/torch.sum(intensity))
            if idx !=3:
                cropped_mask = mask[mask_center_mask,:][:, mask_center_mask]
            else :
                cropped_mask = mask[mask_center_asm,:][:, mask_center_asm]
            
            cropped_intensities.append(cropped_intensity)
            cropped_masks.append(cropped_mask)

        list_cropped_target_fields = []
        list_cropped_target_fields_asm = []
        column_titles = ["SLM + Lens", "Phase inversion + Lens", "Phase inversion + dithering + Lens", "Phase inversion + dithering + Fresnel Lens", "Optim. with lens","Optim. with lens + quantization"]

        for idx, target_field in enumerate(list_target_fields):
            list_cropped_target_fields.append(target_field[mask_CRIGF_y, :][:, mask_CRIGF_x])
        
        for idx, target_field in enumerate(list_target_fields_asm):
            list_cropped_target_fields_asm.append(target_field[mask_CRIGF_asm_y, :][:, mask_CRIGF_asm_x])


        cross_overlap_integrals = []
        energies_inside = []

        for idx in range(len(cropped_amplitudes)):
            if idx !=3:
                cross_overlap_integral = compute_loss_no_weights(cropped_amplitudes[idx], list_cropped_target_fields)
            else :
                cross_overlap_integral = compute_loss_no_weights(cropped_amplitudes[idx], list_cropped_target_fields_asm)
                
            cross_overlap_integrals.append(cross_overlap_integral)
            energies_inside.append(cropped_energy_inside[idx] * 100)


            list_cross_correlation_matrix[idx][mode_nb,:] = np.array(cross_overlap_integrals[idx][0:nb_of_modes_to_consider])

        fig, ax = plt.subplots(2, 6, figsize=(32, 10))  # Change to 4 columns

        for i, title in enumerate(column_titles):
            ax[0, i].set_title(title)

        for i in range(6):  # Change to range(4)
            print(cross_overlap_integrals[i])
            im1 = ax[0, i].imshow(cropped_masks[i],
                                  extent=[-crop_size_detector / (2 * mm), crop_size_detector / (2 * mm),
                                          -crop_size_detector / (2 * mm), crop_size_detector / (2 * mm)], cmap="twilight",
                                  vmin=-np.pi, vmax=np.pi)
            ax[0, i].set_xlabel("X [mm]")
            if i == 0:
                ax[0, i].set_ylabel("Y [mm]")

            im2 = ax[1, i].imshow(cropped_intensities[i].detach().numpy(),
                                  extent=[-crop_size_detector / (2 * um), crop_size_detector / (2 * um),
                                          -crop_size_detector / (2 * um), crop_size_detector / (2 * um)], cmap="gray",
                                  vmin=0, vmax=4000)
            ax[1, i].set_xlabel("X [um]")
            if i == 0:
                ax[1, i].set_ylabel("Y [um]")

            # Overlay energy and overlap integral
            energy_text = f"Energy inside: {energies_inside[i]:.2f}%"
            overlap_text = f"Overlap integral: {cross_overlap_integrals[i][mode_nb]:.2f}"
            
            # Add overlap integral text to bottom left
            ax[1, i].text(0.05, 0.05, overlap_text, transform=ax[1, i].transAxes, color='white',
                          fontsize=12, verticalalignment='bottom', horizontalalignment='left',
                          bbox=dict(facecolor='black', alpha=0.5))
            
            # Add energy inside text to top right
            ax[1, i].text(0.95, 0.95, energy_text, transform=ax[1, i].transAxes, color='white',
                          fontsize=12, verticalalignment='top', horizontalalignment='right',
                          bbox=dict(facecolor='black', alpha=0.5))

        cbar1 = fig.colorbar(im1, ax=ax[0, -1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.set_label('Phase [radians]')
        cbar2 = fig.colorbar(im2, ax=ax[1, -1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.set_label('Intensity [a.u.]')
        plt.tight_layout()
        plt.savefig(dir_path + f"maps_{mode_nb}.svg")
        plt.savefig(dir_path + f"maps_{mode_nb}.png")
        # plt.show()

    # Calculate the global vmin and vmax
    all_values = np.concatenate([matrix.flatten() for matrix in list_cross_correlation_matrix])
    global_vmin = all_values.min()+1e-10
    global_vmax = all_values.max()

    fig, ax = plt.subplots(1, 6, figsize=(32, 5))  # Change to 4 columns

    for i, title in enumerate(column_titles):
        ax[i].set_title(title)
        im = ax[i].imshow(list_cross_correlation_matrix[i], cmap=new_cmap,
                          norm=LogNorm(vmin=global_vmin, vmax=global_vmax))
        ax[i].set_ylabel("Target Mode")
        ax[i].set_xlabel("Mode Number")

        # Add the text annotations for each matrix value
        for (j, k), val in np.ndenumerate(list_cross_correlation_matrix[i]):
            ax[i].text(k, j, format_val(val), ha='center', va='center', color='white', fontsize=7,
                       bbox=dict(facecolor='black', alpha=0.2, edgecolor='none'))

        cbar1 = fig.colorbar(im, ax=ax[i], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.set_label('Overlap Integral')

    plt.tight_layout()
    plt.savefig(dir_path + "cross_correlation_matrix.svg")
    plt.savefig(dir_path + "cross_correlation_matrix.png")


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)
