import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from units import *
from helpers import load_yaml_config
from simulation import Simulation
from sources import Source
from slm import SLM
from ft_lens import FT_Lens, propagate_and_collect_side_view
from FieldGeneration import generate_target_profiles_specific_modes,generate_target_profile_CRIGF
from cost_functions import calculate_normalized_overlap
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import math
from tqdm import tqdm
import csv
from tabulate import tabulate
import sys

class OpticalSystem(nn.Module):
    def __init__(self, device="cpu", target_mode_nbs=(2,2), with_minimize_losses=True):
        super(OpticalSystem, self).__init__()

        # Load configurations
        self.config_source = load_yaml_config("./configs/source.yml")
        self.config_slm = load_yaml_config("./configs/SLM.yml")
        self.config_simulation = load_yaml_config("./configs/simulation.yml")

        # Initialize components
        self.simulation = Simulation(config_dict=self.config_simulation)
        self.source = Source(config_dict=self.config_source, XY_grid=self.simulation.XY_grid)
        self.ft_lens = FT_Lens(self.simulation.delta_x_in, self.simulation.XY_grid, self.source.wavelength)

        # Set device
        self.device = torch.device(device)

        # Prepare target field
        self.target_modes = target_mode_nbs

        self.with_minimize_losses = with_minimize_losses
        self.list_target_files_0 = generate_target_profiles_specific_modes(yaml_file="./configs/target_profile.yml",
                                                     XY_grid=self.ft_lens.XY_output_grid,
                                                     list_modes_nb=[0,1,2,3,4,5,6,7],orientation="vertical")
        self.list_target_files_1 = generate_target_profiles_specific_modes(yaml_file="./configs/target_profile.yml",
                                                     XY_grid=self.ft_lens.XY_output_grid,
                                                     list_modes_nb=[0,1,2,3,4,5,6,7],orientation="horizontal")
        
        self.CRIGF_shape = generate_target_profile_CRIGF(list_mode_nb=target_mode_nbs,XY_grid=self.ft_lens.XY_output_grid).to(self.device)


        # Move target fields to device
        self.list_target_files_horizontal = [field.to(self.device) for field in self.list_target_files_0]
        self.list_target_fields_horizontal_sum = [torch.sum(field,dim=0) for field in self.list_target_files_horizontal]

        self.list_target_files_vertical = [field.to(self.device) for field in self.list_target_files_1]
        self.list_target_fields_vertical_sum = [torch.sum(field,dim=1) for field in self.list_target_files_vertical]


        self.target_field_horizontal = self.list_target_files_horizontal[self.target_modes[0]]
        self.target_field_vertical = self.list_target_files_vertical[self.target_modes[1]]

        self.slm = SLM(config_dict=self.config_slm, XY_grid=self.simulation.XY_grid)

        # Replace the hypergaussian_weights call with the new method
        self.weights_map = self.create_target_based_weights(
            self.CRIGF_shape,
            sigma=2  # Adjust this value to control the Gaussian blur
        ).to(self.device)


    def forward(self, source_field,epoch):
        # Apply the SLM phase modulation
        modulated_field = self.slm.apply_phase_modulation_sigmoid(source_field,steepness=2 + epoch/200,num_terms=1, spacing=1)

        # Propagate the field
        out_field = self.ft_lens(modulated_field, pad=False, flag_ifft=False)

        return out_field

    def create_target_based_weights(self, target_field, sigma=2):
        # Create binary mask based on non-zero amplitudes of the target field
        binary_mask = (torch.abs(target_field) > 0).float()

        # Create Gaussian kernel for convolution
        kernel_size = int(6 * sigma + 1)  # Ensure odd kernel size
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        gaussian_kernel = self.create_gaussian_kernel(kernel_size, sigma).to(self.device)

        # Convolve binary mask with Gaussian kernel
        weights_map = nn.functional.conv2d(
            binary_mask.unsqueeze(0).unsqueeze(0),
            gaussian_kernel.unsqueeze(0).unsqueeze(0),
            padding=kernel_size // 2
        ).squeeze()

        # Normalize weights to [0, 1] range
        weights_map = (weights_map - weights_map.min()) / (weights_map.max() - weights_map.min())

        return weights_map

    def create_gaussian_kernel(self, kernel_size, sigma):
        x = torch.arange(kernel_size, device=self.device) - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel.outer(kernel)

    def compute_loss(self, out_field, list_target_field_horizontal, list_target_field_vertical, epoch):
        # Apply the weight map to the fields
        weighted_out_field = out_field * self.weights_map

        # Calculate total energy of the weighted output field
        total_weighted_energy = torch.sum(torch.abs(weighted_out_field) ** 2)

        # Find rows and columns where weights_map > 1e-3
        rows_to_consider = torch.any(self.weights_map > 1e-3, dim=1)
        cols_to_consider = torch.any(self.weights_map > 1e-3, dim=0)

        list_overlaps_horizontal = []
        list_overlaps_vertical = []

        for target_field_horizontal, target_field_vertical in zip(list_target_field_horizontal, list_target_field_vertical):
            # Horizontal overlap calculation
            out_rows = weighted_out_field[:, cols_to_consider]
            target_rows = target_field_horizontal[cols_to_consider].unsqueeze(0).expand_as(out_rows)
            row_energies = torch.sum(torch.abs(out_rows) ** 2, dim=1)
            overlaps_h = calculate_normalized_overlap(out_rows, target_rows)
            weighted_overlap_horizontal = torch.sum(overlaps_h * row_energies)

            # Vertical overlap calculation
            out_cols = weighted_out_field[rows_to_consider, :]
            target_cols = target_field_vertical[rows_to_consider].unsqueeze(1).expand_as(out_cols)
            col_energies = torch.sum(torch.abs(out_cols) ** 2, dim=0)
            overlaps_v = calculate_normalized_overlap(out_cols, target_cols)
            weighted_overlap_vertical = torch.sum(overlaps_v * col_energies)

            # Normalize the weighted overlaps
            normalized_overlap_horizontal = weighted_overlap_horizontal / total_weighted_energy
            normalized_overlap_vertical = weighted_overlap_vertical / total_weighted_energy

            list_overlaps_horizontal.append(normalized_overlap_horizontal)
            list_overlaps_vertical.append(normalized_overlap_vertical)

        loss_horizontal = -torch.log(list_overlaps_horizontal[self.target_modes[0]] + 1e-10) * 1e-1
        loss_vertical = -torch.log(list_overlaps_vertical[self.target_modes[1]] + 1e-10) * 1e-1

        self.target_mode_horizontal = list_overlaps_horizontal[self.target_modes[0]]
        self.target_mode_vertical = list_overlaps_vertical[self.target_modes[1]]

        # Exclude the target mode from the overlaps
        other_overlaps_horizontal = [overlap for idx, overlap in enumerate(list_overlaps_horizontal) if idx != self.target_modes[0]]
        other_overlaps_vertical = [overlap for idx, overlap in enumerate(list_overlaps_vertical) if idx != self.target_modes[1]]

        max_other_overlap_horizontal = torch.max(torch.stack(other_overlaps_horizontal))
        max_other_overlap_vertical = torch.max(torch.stack(other_overlaps_vertical))

        loss_others_horizontal = -epoch * torch.log(1 - max_other_overlap_horizontal + 1e-10)*1e-1
        loss_others_vertical = -epoch * torch.log(1 - max_other_overlap_vertical + 1e-10)*1e-1

        # Total Variation (TV) regularization for smoothness
        diff_dim0 = torch.diff(self.slm.phase, dim=0) ** 2
        diff_dim1 = torch.diff(self.slm.phase, dim=1) ** 2

        # Pad the differences to match original size
        diff_dim0_padded = nn.functional.pad(diff_dim0, (0, 0, 0, 1), 'constant', 0)
        diff_dim1_padded = nn.functional.pad(diff_dim1, (0, 1, 0, 0), 'constant', 0)

        TV_term = torch.sum(torch.sqrt(diff_dim0_padded + diff_dim1_padded + 1e-2))
        loss_tv = TV_term * 5e-6 * (50 / (epoch + 1))


        loss_energy = -1*total_weighted_energy/torch.sum(torch.abs(out_field) ** 2)*1e-1

        loss = loss_horizontal + loss_vertical + loss_others_horizontal + loss_others_vertical + loss_tv + loss_energy

        # Collect loss components for logging
        loss_components = {
            'total_loss': loss.item(),
            'loss_horizontal': loss_horizontal.item(),
            'loss_vertical': loss_vertical.item(),
            'loss_others_horizontal': loss_others_horizontal.item(),
            'loss_others_vertical': loss_others_vertical.item(),
            'loss_tv': loss_tv.item(),
            'loss_energy': loss_energy.item(),
            'inside_energy_percentage': total_weighted_energy.item() / torch.sum(torch.abs(out_field) ** 2).item(),
            'overlaps_horizontal': [overlap.item()*100 for overlap in list_overlaps_horizontal],
            'overlaps_vertical': [overlap.item()*100 for overlap in list_overlaps_vertical]
        }

        return loss, loss_components

def normalize_image(tensor):
    tensor = tensor.detach().cpu().numpy()
    tensor = np.abs(tensor)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    return tensor

def optimize_phase_mask(target_mode_nbs, run_name, run_number, data_dir, num_epochs, pbar):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the TensorBoard writer with a unique run name
    writer = SummaryWriter(log_dir=f'{data_dir}/{run_name}/logs/mode_{target_mode_nbs}/run_{run_number}')

    # Initialize the model
    model = OpticalSystem(device=device, target_mode_nbs=target_mode_nbs, with_minimize_losses=True)

    # Move source field to device
    source_field = model.source.field.field.to(model.device)

    # Prepare the optimizer
    optimizer = optim.Adam([model.slm.phase_parameters], lr=0.1)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-2)

    save_mask_epochs = []

    # Calculate epochs to save masks, starting from 100
    for i in range(int(math.log2(num_epochs)) + 1):
        save_mask_epochs.extend(range(max(2**i - 1, 100), 2**(i+1) - 1, 2**i))
    save_mask_epochs = [epoch for epoch in save_mask_epochs if 100 <= epoch < num_epochs]

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        out_field = model(source_field, epoch)

        # Compute loss
        loss, loss_components = model.compute_loss(out_field, model.list_target_fields_horizontal_sum, 
                                                              model.list_target_fields_vertical_sum, epoch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        global_step = epoch

        # Log scalars with more descriptive names
        writer.add_scalar('Loss/Total', loss_components['total_loss'], global_step)
        writer.add_scalar('Loss/Horizontal', loss_components['loss_horizontal'], global_step)
        writer.add_scalar('Loss/Vertical', loss_components['loss_vertical'], global_step)
        writer.add_scalar('Loss/Others_Horizontal', loss_components['loss_others_horizontal'], global_step)
        writer.add_scalar('Loss/Others_Vertical', loss_components['loss_others_vertical'], global_step)
        writer.add_scalar('Loss/TV', loss_components['loss_tv'], global_step)
        writer.add_scalar('Loss/Energy', loss_components['loss_energy'], global_step)
        writer.add_scalar('Metrics/Inside Energy Percentage', loss_components['inside_energy_percentage'], global_step)

        # Log overlaps for horizontal and vertical modes
        for idx, overlap in enumerate(loss_components['overlaps_horizontal']):
            writer.add_scalar(f'Overlap/Horizontal Mode {idx}', overlap, global_step)
        for idx, overlap in enumerate(loss_components['overlaps_vertical']):
            writer.add_scalar(f'Overlap/Vertical Mode {idx}', overlap, global_step)

        # Log learning rate
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step)

        # Update progress bar description with current loss and learning rate
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'})
        pbar.update(1)

        if epoch in save_mask_epochs:
            # Save the phase mask
            os.makedirs(f'{data_dir}/{run_name}/phase_masks/mode_{target_mode_nbs}/run_{run_number}', exist_ok=True)
            np.save(f'{data_dir}/{run_name}/phase_masks/mode_{target_mode_nbs}/run_{run_number}/slm_phase_epoch_{epoch}.npy', model.slm.phase.detach().cpu().numpy())

            # Log images
            # Phase mask
            # phase_mask = model.slm.phase.detach().cpu().numpy()
            # phase_mask = (phase_mask - phase_mask.min()) / (phase_mask.max() - phase_mask.min() + 1e-8)
            # phase_mask = phase_mask[np.newaxis, :, :]  # Add channel dimension
            # writer.add_image('SLM/phase_mask', phase_mask, global_step)

            # # Output field intensity
            # output_intensity = normalize_image(out_field)
            # output_intensity = output_intensity[np.newaxis, :, :]
            # writer.add_image('Field/output_intensity', output_intensity, global_step)

            # #cut horizontal and vertical
            # out_field_horizontal = model.out_field_horizontal
            # out_field_vertical = model.out_field_vertical
            # out_target_horizontal = model.target_mode_horizontal
            # out_target_vertical = model.target_mode_vertical
            

            # Target field intensity
            # target_intensity = normalize_image(model.target_field_horizontal)
            # target_intensity = target_intensity[np.newaxis, :, :]
            # writer.add_image('Field/target_intensity_horizontal', target_intensity, global_step)

            # target_intensity = normalize_image(model.target_field_vertical)
            # target_intensity = target_intensity[np.newaxis, :, :]
            # writer.add_image('Field/target_intensity_vertical', target_intensity, global_step)
            # # Optional: Log side-view propagation figure
            # fig = plot_side_view(model, source_field)
            # writer.add_figure('Propagation/side_view', fig, global_step)
            # plt.close(fig)

            # Compare 1D plots
            # out_field_horizontal = model.out_field_horizontal.detach().cpu().numpy()
            # out_field_vertical = model.out_field_vertical.detach().cpu().numpy()
            # target_field_horizontal = model.target_field_horizontal.sum(dim=0).detach().cpu().numpy()
            # target_field_vertical = model.target_field_vertical.sum(dim=1).detach().cpu().numpy()

            # # Normalize the fields
            # out_field_horizontal = np.abs(out_field_horizontal) / np.max(np.abs(out_field_horizontal))
            # out_field_vertical = np.abs(out_field_vertical) / np.max(np.abs(out_field_vertical))
            # target_field_horizontal = np.abs(target_field_horizontal) / np.max(np.abs(target_field_horizontal))
            # target_field_vertical = np.abs(target_field_vertical) / np.max(np.abs(target_field_vertical))

            # # Create comparison plots
            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

            # # Horizontal comparison
            # ax1.plot(out_field_horizontal, label='Output')
            # ax1.plot(target_field_horizontal, label='Target')
            # ax1.set_xlim(800,1200)
            # ax1.set_title(f'Horizontal Comparison (Epoch {epoch})')
            # ax1.legend()
            # ax1.set_xlabel('Position')
            # ax1.set_ylabel('Normalized Intensity')

            # # Vertical comparison
            # ax2.plot(out_field_vertical, label='Output')
            # ax2.plot(target_field_vertical, label='Target')
            # ax2.set_xlim(800,1200)
            # ax2.set_title(f'Vertical Comparison (Epoch {epoch})')
            # ax2.legend()
            # ax2.set_xlabel('Position')
            # ax2.set_ylabel('Normalized Intensity')

            # plt.tight_layout()
            # plt.close(fig)

            # # Log the comparison plot to TensorBoard
            # writer.add_figure('Comparison/1D_plots', fig, global_step)

    # Return the final results
    final_overlaps_horizontal = loss_components['overlaps_horizontal']
    final_overlaps_vertical = loss_components['overlaps_vertical']
    final_inside_energy = loss_components['inside_energy_percentage'] * 100
    return final_overlaps_horizontal, final_overlaps_vertical, final_inside_energy

def run_multiple_tests(data_dir, target_modes, num_runs_per_mode, num_epochs, run_name=None):
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the main directory for this run
    run_dir = f"{data_dir}/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    results = []
    headers = ["Target Mode", "Run"] + [f"H{i}" for i in range(8)] + [f"V{i}" for i in range(8)] + ["Inside Energy %"]

    # Create the CSV file and write the headers
    csv_filename = f"{run_dir}/results.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

    # Create a progress bar for the overall process
    total_iterations = len(target_modes) * num_runs_per_mode
    overall_pbar = tqdm(total=total_iterations, desc="Overall Progress", position=0, leave=True)

    for mode in target_modes:
        for run in range(num_runs_per_mode):
            # Create a nested progress bar for the current optimization
            optimize_pbar = tqdm(total=num_epochs, desc=f"Mode {mode}, Run {run+1}", position=1, leave=False)
            
            # Pass the progress bar to optimize_phase_mask
            overlaps_h, overlaps_v, inside_energy = optimize_phase_mask(target_mode_nbs=mode, run_name=run_name, run_number=run + 1, data_dir=data_dir, num_epochs=num_epochs, pbar=optimize_pbar)
            result = [mode, run + 1] + overlaps_h + overlaps_v + [inside_energy]
            results.append(result)

            # Append the result to the CSV file
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(result)

            # Close the nested progress bar
            optimize_pbar.close()

            # Clear the console and print the updated table
            os.system('cls' if os.name == 'nt' else 'clear')
            table = tabulate(results, headers=headers, floatfmt=".2f", tablefmt="plain", numalign="right")
            print(table)
            print("\n")  # Add some space after the table
            
            # Update the overall progress bar
            overall_pbar.update(1)

    overall_pbar.close()
    
    print(f"\nResults saved to {csv_filename}")




# In the optimize_phase_mask function, after initializing the model:

if __name__ == "__main__":

    data_dir = "/data/arouxel"  # You can change this to any directory you want
    target_modes = [(0,1),(1,1),(1,2),(2,2),(3,3),(3,2), (2,3), (5,3), (3,4),(6,7),(7,6)]  # Add or remove modes as needed
    # target_modes = [(1,2)]
    num_runs_per_mode = 5
    num_epochs = 8500  # Set the number of epochs here
    run_name = "Phase_masks_2D_with_lens_here_we_go_again"  # Optional: provide a custom run name

    run_multiple_tests(data_dir, target_modes, num_runs_per_mode, num_epochs,run_name)





