import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from simulation import Simulation
from sources import Source
from slm import SLM
from ft_lens import FT_Lens, ASMPropagation, create_binary_fresnel_lens
from units import *
from helpers import load_yaml_config, generate_target_amplitude, design_mask, generate_target_mask, wrap_phase
from FieldGeneration import generate_target_profiles
from cost_functions import calculate_normalized_overlap

def floyd_steinberg(image):
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[y, x]
            new = np.round(old)
            image[y, x] = new
            error = old - new
            if x + 1 < w:
                image[y, x + 1] += error * 0.4375
            if (y + 1 < h) and (x + 1 < w):
                image[y + 1, x + 1] += error * 0.0625
            if y + 1 < h:
                image[y + 1, x] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < h):
                image[y + 1, x - 1] += error * 0.1875
    return image

def compute_loss_no_weights(out_field, list_target_field):
    return [float(calculate_normalized_overlap(out_field, target_field)) for target_field in list_target_field]

def main():
    # Load configurations
    config_source = load_yaml_config("./configs/source.yml")
    config_slm = load_yaml_config("./configs/SLM.yml")
    config_simulation = load_yaml_config("./configs/simulation.yml")
    config_simulation_asm = load_yaml_config("./configs/simulation_ASM.yml")

    # Set up simulations
    simulation = Simulation(config_dict=config_simulation)
    simulation_asm = Simulation(config_dict=config_simulation_asm)

    # Set up sources
    source = Source(config_dict=config_source, XY_grid=simulation.XY_grid)
    source_asm = Source(config_dict=config_source, XY_grid=simulation_asm.XY_grid)

    # Set up propagation methods
    ft_lens = FT_Lens(simulation.delta_x_in, simulation.XY_grid, source.wavelength)
    asm = ASMPropagation(simulation_asm.delta_x_in, source_asm.wavelength, 20*mm, simulation_asm.XY_grid[0].shape)

    # Generate target profiles
    list_target_fields = generate_target_profiles(yaml_file="./configs/target_profile.yml",
                                                  XY_grid=ft_lens.XY_output_grid,
                                                  list_modes_nb=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    list_target_fields_asm = generate_target_profiles(yaml_file="./configs/target_profile.yml",
                                                      XY_grid=simulation_asm.XY_grid,
                                                      list_modes_nb=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    nb_of_modes_to_consider = 9
    cross_correlation_matrices = [np.zeros((nb_of_modes_to_consider, nb_of_modes_to_consider)) for _ in range(3)]

    # Define crop sizes and masks
    crop_size_detector = 100 * um
    crop_size_slm = 5 * mm
    position = 0.12 * mm
    height = 60 * um  # Make sure this is defined earlier in your code

    # Define masks for different approaches
    mask_with_center = (ft_lens.XY_output_grid[1][0, :] > -crop_size_detector / 2) & (ft_lens.XY_output_grid[1][0, :] < crop_size_detector / 2)
    mask_with_wedge_x = (ft_lens.XY_output_grid[1][0, :] > position - crop_size_detector / 2) & (ft_lens.XY_output_grid[1][0, :] < position + crop_size_detector / 2)
    mask_with_wedge_y = (ft_lens.XY_output_grid[1][0, :] > -crop_size_detector / 2) & (ft_lens.XY_output_grid[1][0, :] < crop_size_detector / 2)
    mask_center_mask = (simulation.XY_grid[1][0,:] > -crop_size_slm / 2) & (simulation.XY_grid[1][0,:] < crop_size_slm / 2)
    mask_CRIGF = (ft_lens.XY_output_grid[1][0, :] > -height/2) & (ft_lens.XY_output_grid[1][0, :] < height/2)
    mask_CRIGF_switch_x = (ft_lens.XY_output_grid[1][0, :] > position-height/2) & (ft_lens.XY_output_grid[1][0, :] < position+height/2)
    mask_CRIGF_switch_y = (ft_lens.XY_output_grid[1][0, :] > -height/2) & (ft_lens.XY_output_grid[1][0, :] < height/2)


    # Crop target fields
    list_cropped_target_fields = []
    for target_field in list_target_fields:
        list_cropped_target_fields.append(target_field[mask_CRIGF, :][:, mask_CRIGF])

    # Crop target fields for ASM
    mask_CRIGF_asm = (simulation_asm.XY_grid[1][0, :] > -height/2) & (simulation_asm.XY_grid[1][0, :] < height/2)

    list_cropped_target_fields_asm = []
    for target_field in list_target_fields_asm:
        list_cropped_target_fields_asm.append(target_field[mask_CRIGF_asm, :][:, mask_CRIGF_asm])

    for mode_nb in range(nb_of_modes_to_consider):
        print(f"Processing mode {mode_nb}")

        # Generate target amplitude
        width = 52 * um
        height = 52 * um
        sinus_period = 2 * width / (mode_nb + 1)
        phase_off = np.pi / 2 if mode_nb % 2 == 0 else 0

        target_amplitude = generate_target_amplitude(simulation.XY_grid, ft_lens.XY_output_grid, source.wavelength,
                                                     ft_lens.focal_length, amplitude_type="Rectangle", width=width, height=width)
        target_amplitude *= generate_target_amplitude(simulation.XY_grid, ft_lens.XY_output_grid, source.wavelength,
                                                      ft_lens.focal_length, amplitude_type="Sinus", period=sinus_period, phase_offset=phase_off)

        inverse_fourier_transform = ft_lens(target_amplitude, pad=False, flag_ifft=True)

        # Generate masks
        wedge_mask = design_mask(simulation.XY_grid, "Wedge", source.wavelength, ft_lens.focal_length, angle=0, position=0.12 * mm)
        phase_inversion_mask = generate_target_mask(inverse_fourier_transform, mask_type="phase target field")
        amplitude_modulation_mask, _, _ = generate_target_mask(inverse_fourier_transform, mask_type="modulation amplitude", input_field=source.field.field)

        # 1. Davis et al. 1999 (FT lens)
        mask_davis = wrap_phase(phase_inversion_mask + wedge_mask) * amplitude_modulation_mask

        # 2. Phase inversion + dithering (FT lens)
        target_field_unnormalized = torch.real(inverse_fourier_transform) if mode_nb % 2 == 0 else torch.imag(inverse_fourier_transform)
        new_mask = (target_field_unnormalized - torch.min(target_field_unnormalized)) / (torch.max(target_field_unnormalized) - torch.min(target_field_unnormalized))
        target_field_unnormalized /= (torch.max(target_field_unnormalized) * 2)
        new_mask = target_field_unnormalized + 0.5
        dithered_amplitude = torch.tensor(floyd_steinberg(new_mask.numpy()))
        mask_dithered = dithered_amplitude * np.pi - np.pi / 2

        # 3. Phase inversion + dithering + Fresnel lens (ASM propagation)
        fresnel_mask, _ = create_binary_fresnel_lens(simulation_asm.XY_grid[0].shape,
                                                     (simulation_asm.delta_x_in, simulation_asm.delta_x_in),
                                                     source_asm.wavelength, 20*mm, radius=1.5*mm)
        mask_combined = torch.angle(fresnel_mask * torch.exp(1j * mask_dithered))

        masks = [mask_davis, mask_dithered, mask_combined]
        simulations = [simulation, simulation, simulation_asm]
        sources = [source, source, source_asm]
        propagations = [ft_lens, ft_lens, asm]

        # Create the first figure: masks and cropped output fields
        fig1, ax1 = plt.subplots(2, 3, figsize=(15, 10))
        titles = ["Davis et al. 1999 (FT lens)", "Phase inversion + dithering (FT lens)", "Phase inversion + dithering + Fresnel lens (ASM)"]

        for i in range(3):
            # Plot mask
            ax1[0, i].imshow(masks[i], cmap='twilight', vmin=-np.pi, vmax=np.pi)
            ax1[0, i].set_title(titles[i])
            ax1[0, i].axis('off')
            slm = SLM(config_dict=config_slm, XY_grid=simulation.XY_grid, initial_phase=masks[i],device='cpu')
            # Plot cropped output field
            output_field = propagations[i](slm.apply_phase_modulation(sources[i].field.field, mapping=False))
            if i == 0:  # Davis approach
                cropped_field = output_field[mask_CRIGF_switch_y][:, mask_CRIGF_switch_x]
                y_start, y_end = np.where(mask_CRIGF_switch_y)[0][[0, -1]]
                x_start, x_end = np.where(mask_CRIGF_switch_x)[0][[0, -1]]
            elif i == 2:  # ASM approach
                cropped_field = output_field[mask_CRIGF_asm][:, mask_CRIGF_asm]
                y_start, y_end = np.where(mask_CRIGF_asm)[0][[0, -1]]
                x_start, x_end = y_start, y_end  # Assuming it's square
            else:  # Other approaches
                cropped_field = output_field[mask_CRIGF][:, mask_CRIGF]
                y_start, y_end = np.where(mask_CRIGF)[0][[0, -1]]
                x_start, x_end = y_start, y_end  # Assuming it's square

            # Extend the cropping area by 10 pixels on each side
            y_start, y_end = max(0, y_start - 10), min(output_field.shape[0], y_end + 10)
            x_start, x_end = max(0, x_start - 10), min(output_field.shape[1], x_end + 10)

            extended_field = output_field[y_start:y_end, x_start:x_end]
            
            intensity = torch.abs(extended_field)**2
            ax1[1, i].imshow(intensity.detach().cpu().numpy(), cmap='hot')
            ax1[1, i].axis('off')

            # Add energy inside information
            total_power = torch.sum(torch.abs(output_field)**2)
            power_inside = torch.sum(intensity)
            energy_inside = (power_inside / total_power) * 100
            energy_text = f"Energy inside: {energy_inside:.2f}%"
            ax1[1, i].text(0.05, 0.95, energy_text, transform=ax1[1, i].transAxes, color='white', fontsize=8,
                           verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f"comparison_results_mode_{mode_nb}.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # Update cross-correlation matrices
        for i in range(3):
            slm = SLM(config_dict=config_slm, XY_grid=simulation.XY_grid, initial_phase=masks[i],device='cpu')
            output_field = propagations[i](slm.apply_phase_modulation(sources[i].field.field, mapping=False))
            if i == 0:  # Davis approach
                cropped_field = output_field[mask_CRIGF_switch_y][:, mask_CRIGF_switch_x]
            elif i == 2:  # ASM approach
                cropped_field = output_field[mask_CRIGF_asm][:, mask_CRIGF_asm]
            else:  # Other approaches
                cropped_field = output_field[mask_CRIGF][:, mask_CRIGF]
            if i == 2:  # ASM approach
                target_fields = list_cropped_target_fields_asm
            else:
                target_fields = list_cropped_target_fields

            fig,ax = plt.subplots(1,2)
            cross_overlap_integral = compute_loss_no_weights(cropped_field, target_fields)
            cross_correlation_matrices[i][mode_nb, :] = cross_overlap_integral[:nb_of_modes_to_consider]

    # Create the second figure: cross-correlation matrices
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Davis et al. 1999 (FT lens)", "Phase inversion + dithering (FT lens)", "Phase inversion + dithering + Fresnel lens (ASM)"]

    for idx, (ax, matrix, title) in enumerate(zip(axes, cross_correlation_matrices, titles)):
        im = ax.imshow(matrix, cmap='hot', norm=LogNorm(vmin=1e-3, vmax=1))
        ax.set_title(title)
        ax.set_xlabel("Target Mode")
        ax.set_ylabel("Input Mode")
        plt.colorbar(im, ax=ax, label="Overlap Integral")

        for i in range(nb_of_modes_to_consider):
            for j in range(nb_of_modes_to_consider):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)

    plt.tight_layout()
    plt.savefig("comparison_cross_correlation_matrices.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)

if __name__ == "__main__":
    main()