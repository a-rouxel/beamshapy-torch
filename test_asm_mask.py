import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from helpers import load_yaml_config
from simulation import Simulation
from sources import Source
from slm import SLM
from ft_lens import ASMPropagation
from units import *
from FieldGeneration import generate_target_profiles
from cost_functions import calculate_normalized_overlap, quantize_phase

def format_val(val):
    """Format the value to scientific notation if smaller than 0.005, else to two decimal places."""
    return f'{val:.1e}' if val < 0.005 else f'{val:.2f}'

def compute_loss_no_weights(out_field, list_target_field):
    list_overlaps = []
    for target_field in list_target_field:
        overlap = calculate_normalized_overlap(out_field, target_field)
        list_overlaps.append(float(overlap))
    return list_overlaps

def calculate_intensity_percentage(intensity, mask):
    """Calculate the percentage of intensity inside a given mask."""
    total_intensity = torch.sum(intensity)
    masked_intensity = torch.sum(intensity[mask])
    return (masked_intensity / total_intensity * 100).item()

def main():
    dir_path = "./asm_test_results/"
    os.makedirs(dir_path, exist_ok=True)

    nb_of_modes_to_consider = 9
    cross_correlation_matrix_asm = np.zeros((nb_of_modes_to_consider, nb_of_modes_to_consider))

    config_source = load_yaml_config("./configs/source.yml")
    config_slm = load_yaml_config("./configs/SLM.yml")
    config_simulation_asm = load_yaml_config("./configs/simulation_ASM.yml")

    simulation_asm = Simulation(config_dict=config_simulation_asm)
    source_asm = Source(config_dict=config_source, XY_grid=simulation_asm.XY_grid)
    asm = ASMPropagation(simulation_asm.delta_x_in, source_asm.wavelength, 20*mm, simulation_asm.XY_grid[0].shape)

    list_target_fields_asm = generate_target_profiles(yaml_file="./configs/target_profile.yml",
                                                      XY_grid=simulation_asm.XY_grid,
                                                      list_modes_nb=range(nb_of_modes_to_consider))

    # Parameters
    crop_size_detector = 100 * um
    height = 52 * um

    # Masks
    mask_with_center_asm = (simulation_asm.XY_grid[1][0, :] > - crop_size_detector / 2) & (simulation_asm.XY_grid[1][0, :] < crop_size_detector / 2)
    mask_CRIGF_asm = (simulation_asm.XY_grid[1][0, :] > -height/2) & (simulation_asm.XY_grid[1][0, :] < height/2)

    # Crop the target fields
    list_cropped_target_fields_asm = []
    for target_field in list_target_fields_asm:
        list_cropped_target_fields_asm.append(target_field[mask_CRIGF_asm, :][:, mask_CRIGF_asm])

    for mode_nb in range(nb_of_modes_to_consider):
        print(f"Processing mode {mode_nb}")

        mode_parity = "even" if mode_nb % 2 == 0 else "odd"

        # Load the ASM mask
        mask_5 = np.load(f"/home/arouxel/Documents/POSTDOC/beamshapy-pytorch/lightning_logs/slm_phase_{mode_nb}_0_min_losses_False_epoch_2750.npy")
        mask_5 = torch.tensor(mask_5.copy(), dtype=torch.float32)
        mask_5 = quantize_phase(mask_5, 2, mode_parity=mode_parity)

        # Apply the mask and propagate
        slm_asm = SLM(config_dict=config_slm, XY_grid=simulation_asm.XY_grid, initial_phase=mask_5)
        modulated_field = slm_asm.apply_phase_modulation(source_asm.field.field, mapping=False)
        out_field = asm(modulated_field)

        intensity = torch.abs(out_field) ** 2

        # Crop the output field
        cropped_intensity = intensity[mask_with_center_asm,:][:, mask_with_center_asm]
        cropped_amplitude = out_field[mask_CRIGF_asm][:, mask_CRIGF_asm]

        # Calculate percentage of intensity inside CRIGF area
        percentage_in_crigf = calculate_intensity_percentage(intensity, mask_CRIGF_asm)

        # Compute overlap integrals
        cross_overlap_integral = compute_loss_no_weights(cropped_amplitude, list_cropped_target_fields_asm)
        cross_correlation_matrix_asm[mode_nb,:] = np.array(cross_overlap_integral)

        # Plot the results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.set_title(f"ASM Mask for Mode {mode_nb}")
        im1 = ax1.imshow(mask_5.detach().numpy(), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax1.set_xlabel("X [pixels]")
        ax1.set_ylabel("Y [pixels]")
        plt.colorbar(im1, ax=ax1, label='Phase [radians]')

        ax2.set_title(f"Output Intensity for Mode {mode_nb}")
        im2 = ax2.imshow(cropped_intensity.detach().numpy(), cmap="viridis")
        ax2.set_xlabel("X [pixels]")
        ax2.set_ylabel("Y [pixels]")
        plt.colorbar(im2, ax=ax2, label='Intensity [a.u.]')

        overlap_text = f"Overlap integral: {cross_overlap_integral[mode_nb]:.4f}"
        percentage_text = f"Intensity in CRIGF: {percentage_in_crigf:.2f}%"
        ax2.text(0.05, 0.95, overlap_text + '\n' + percentage_text, transform=ax2.transAxes, color='white', fontsize=10,
                 verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))

        plt.tight_layout()
        plt.savefig(dir_path + f"asm_mode_{mode_nb}.png")
        plt.close()

    # Plot the cross-correlation matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cross_correlation_matrix_asm, cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Overlap Integral')
    plt.title("Cross-correlation Matrix for ASM")
    plt.xlabel("Target Mode")
    plt.ylabel("Input Mode")

    for (j, k), val in np.ndenumerate(cross_correlation_matrix_asm):
        plt.text(k, j, format_val(val), ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(dir_path + "asm_cross_correlation_matrix.png")
    plt.close()

if __name__ == "__main__":
    main()