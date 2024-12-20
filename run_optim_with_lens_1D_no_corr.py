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
from FieldGeneration import generate_target_profiles_specific_modes, generate_target_profile_CRIGF
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
    def __init__(self, device="cpu", target_mode_nb=2, with_minimize_losses=True):
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
        self.target_mode = target_mode_nb
        if self.target_mode %2 == 0:
            self.mode_parity = "even"
        else:
            self.mode_parity = "odd"
        self.with_minimize_losses = with_minimize_losses

        self.list_target_files_0 = generate_target_profiles_specific_modes(yaml_file="./configs/target_profile.yml",
                                                     XY_grid=self.ft_lens.XY_output_grid,
                                                     list_modes_nb=[0,1,2,3,4,5,6,7],orientation="vertical")

        
        self.CRIGF_shape = generate_target_profile_CRIGF(list_mode_nb=(target_mode_nb,target_mode_nb),XY_grid=self.ft_lens.XY_output_grid).to(self.device)


        self.list_target_files_vertical = [field.to(self.device) for field in self.list_target_files_0]
        self.list_target_fields_vertical_sum = [torch.sum(field,dim=1) for field in self.list_target_files_vertical]


        self.target_field_vertical = self.list_target_files_vertical[self.target_mode]

        self.slm = SLM(config_dict=self.config_slm, XY_grid=self.simulation.XY_grid)

        # Replace the hypergaussian_weights call with the new method
        self.weights_map = self.create_target_based_weights(
            self.CRIGF_shape,
            sigma=2  # Adjust this value to control the Gaussian blur
        ).to(self.device)


    def forward(self, source_field,epoch):
        # Apply the SLM phase modulation
        modulated_field = self.slm.apply_phase_modulation_sigmoid(source_field,steepness=2 + epoch/200,num_terms=1, spacing=1,mode_parity=self.mode_parity)

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

    def compute_loss(self, out_field, list_target_field, epoch):
        # Apply the weight map to the fields
        weighted_out_field = out_field

        # Calculate total energy of the weighted output field
        total_weighted_energy = torch.sum(torch.abs(weighted_out_field) ** 2)
        inside_energy = torch.sum(torch.abs(out_field * self.weights_map) ** 2)

        # Find rows and columns where weights_map > 1e-3
        # cols_to_consider = torch.any(self.weights_map > 1e-1, dim=0)

        # Vectorized overlap calculation
        out_cols = weighted_out_field[:, :]  # [height, selected_cols]
        list_overlaps_vertical = []
        
        for target_field_vertical in list_target_field:
            # Expand target field to match output dimensions
            target_cols = target_field_vertical.unsqueeze(1).expand_as(out_cols)  # [height, selected_cols]
            
            # Calculate normalized overlaps for all columns at once
            numerator = torch.abs(torch.sum(out_cols.conj() * target_cols, dim=0)) ** 2
            denominator = (torch.sum(torch.abs(out_cols) ** 2, dim=0) * 
                         torch.sum(torch.abs(target_cols) ** 2, dim=0))
            array_overlaps_v = numerator / (denominator + 1e-7)
            
            # Calculate column energies and weighted overlap
            col_energies = torch.sum(torch.abs(out_cols) ** 2, dim=0)
            weighted_overlap_vertical = torch.sum(array_overlaps_v * col_energies)
            
            # Normalize the weighted overlap
            normalized_overlap_vertical = weighted_overlap_vertical / total_weighted_energy
            list_overlaps_vertical.append(normalized_overlap_vertical)

        # Convert list to tensor for more efficient operations
        all_overlaps = torch.stack(list_overlaps_vertical)
        
        # Target mode loss
        loss_vertical = -torch.log(all_overlaps[self.target_mode] + 1e-10)
        self.target_mode_vertical = all_overlaps[self.target_mode]

        # Create mask for other modes
        mask = torch.ones_like(all_overlaps, dtype=torch.bool)
        mask[self.target_mode] = False
        max_other_overlap_vertical = torch.max(all_overlaps[mask])

        # Loss for other modes
        loss_others_vertical = -100 * torch.log(1 - max_other_overlap_vertical + 1e-7)

        # Total Variation (TV) regularization for smoothness
        diff_dim0 = torch.diff(self.slm.phase, dim=0) ** 2
        diff_dim1 = torch.diff(self.slm.phase, dim=1) ** 2

        slm_phase = self.slm.phase
        # slm_phase_flip_x = torch.flip(slm_phase, dims=[0])  # Flip along the X axis
        slm_phase_flip_y = torch.flip(slm_phase, dims=[1])  # Flip along the Y axis
        # symmetry_loss_x = torch.mean((slm_phase - slm_phase_flip_x) ** 2)
        symmetry_loss_y = torch.mean((slm_phase - slm_phase_flip_y) ** 2)
        symmetry_loss = (symmetry_loss_y)

        # Pad the differences to match original size
        diff_dim0_padded = nn.functional.pad(diff_dim0, (0, 0, 0, 1), 'constant', 0)
        diff_dim1_padded = nn.functional.pad(diff_dim1, (0, 1, 0, 0), 'constant', 0)

        TV_term = torch.sum(torch.sqrt(diff_dim0_padded + diff_dim1_padded + 1e-2))
        loss_tv = TV_term * 5e-6 * (50 / (epoch*0.01 + 1))

        # Energy loss
        loss_energy = -10 * inside_energy / torch.sum(torch.abs(out_field) ** 2)

        # Combine all losses 
        loss = loss_vertical + loss_others_vertical + loss_tv + loss_energy + symmetry_loss

        # Collect loss components for logging
        loss_components = {
            'total_loss': loss.item(),
            'loss_vertical': loss_vertical.item(),
            'loss_others_vertical': loss_others_vertical.item(),
            'loss_tv': loss_tv.item(),
            'loss_energy': loss_energy.item(),
            'inside_energy_percentage': inside_energy.item() / torch.sum(torch.abs(out_field) ** 2).item(),
            'overlaps_vertical': [overlap.item() for overlap in all_overlaps]
        }

        return loss, loss_components

def normalize_image(tensor):
    tensor = tensor.detach().cpu().numpy()
    tensor = np.abs(tensor)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    return tensor

def optimize_phase_mask(target_mode_nb, run_name, run_number, data_dir, num_epochs, pbar):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the TensorBoard writer with a unique run name
    writer = SummaryWriter(log_dir=f'{data_dir}/{run_name}/logs/mode_{target_mode_nb}/run_{run_number}')

    # Initialize the model
    model = OpticalSystem(device=device, target_mode_nb=target_mode_nb, with_minimize_losses=True)

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
        loss, loss_components = model.compute_loss(out_field, model.list_target_fields_vertical_sum, epoch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        global_step = epoch

        # Log scalars with more descriptive names
        writer.add_scalar('Loss/Total', loss_components['total_loss'], global_step)
        writer.add_scalar('Loss/Vertical', loss_components['loss_vertical'], global_step)
        writer.add_scalar('Loss/Others_Vertical', loss_components['loss_others_vertical'], global_step)
        writer.add_scalar('Loss/TV', loss_components['loss_tv'], global_step)
        writer.add_scalar('Loss/Energy', loss_components['loss_energy'], global_step)
        writer.add_scalar('Metrics/Inside Energy Percentage', loss_components['inside_energy_percentage'], global_step)

        for idx, overlap in enumerate(loss_components['overlaps_vertical']):
            writer.add_scalar(f'Overlap/Vertical Mode {idx}', overlap, global_step)

        # Update progress bar description with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        pbar.update(1)

        if epoch in save_mask_epochs:
            # Save the phase mask
            os.makedirs(f'{data_dir}/{run_name}/phase_masks/mode_{target_mode_nb}/run_{run_number}', exist_ok=True)
            np.save(f'{data_dir}/{run_name}/phase_masks/mode_{target_mode_nb}/run_{run_number}/slm_phase_epoch_{epoch}.npy', model.slm.phase.detach().cpu().numpy())

            # Log images
            # Phase mask
            phase_mask = model.slm.phase.detach().cpu().numpy()
            phase_mask = (phase_mask - phase_mask.min()) / (phase_mask.max() - phase_mask.min() + 1e-8)
            phase_mask = phase_mask[np.newaxis, :, :]  # Add channel dimension
            writer.add_image('SLM/phase_mask', phase_mask, global_step)

            # Output field intensity
            output_intensity = normalize_image(out_field)
            output_intensity = output_intensity[np.newaxis, :, :]
            writer.add_image('Field/output_intensity', output_intensity, global_step)

            # # Target field intensity
            # target_intensity = normalize_image(model.target_field)
            # target_intensity = target_intensity[np.newaxis, :, :]
            # writer.add_image('Field/target_intensity', target_intensity, global_step)

            # # Optional: Log side-view propagation figure
            # fig = plot_side_view(model, source_field)
            # writer.add_figure('Propagation/side_view', fig, global_step)
            # plt.close(fig)

    # Return the final results
    final_overlaps = loss_components['list_overlaps']
    final_inside_energy = loss_components['inside_energy_percentage']*100
    return final_overlaps, final_inside_energy

def run_multiple_tests(data_dir, target_modes, num_runs_per_mode, num_epochs, run_name=None):
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the main directory for this run
    run_dir = f"{data_dir}/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    results = []
    headers = ["Target Mode", "Run"] + [f"Mode {i}" for i in range(9)] + ["Inside Energy %"]

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
            overlaps, inside_energy = optimize_phase_mask(target_mode_nb=mode, run_name=run_name, run_number=run + 1, data_dir=data_dir, num_epochs=num_epochs, pbar=optimize_pbar)
            result = [mode, run + 1] + overlaps + [inside_energy]
            results.append(result)

            # Append the result to the CSV file
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(result)

            # Close the nested progress bar
            optimize_pbar.close()

            # Clear the console and print the updated table
            os.system('cls' if os.name == 'nt' else 'clear')
            table = tabulate(results, headers=headers, floatfmt=".4f", tablefmt="grid")
            print(table)
            print("\n")  # Add some space after the table
            
            # Update the overall progress bar
            overall_pbar.update(1)

    overall_pbar.close()
    
    print(f"\nResults saved to {csv_filename}")

def plot_side_view(model, source_field):
    # Generate z values
    z_values = torch.linspace(0, 15 * mm, 100).to(model.device)

    # Apply the current phase mask
    modulated_field = source_field * torch.exp(1j * model.slm.phase)

    # Collect side-view intensity pattern
    side_view_intensity = propagate_and_collect_side_view(
        modulated_field,
        model.simulation.delta_x_in,
        model.source.wavelength,
        z_values,
        model.simulation.XY_grid[0].shape
    )

    # Convert to numpy
    side_view_intensity = side_view_intensity.detach().cpu().numpy()

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    extent = [
        z_values[0].item(),
        z_values[-1].item(),
        -model.simulation.XY_grid[0].shape[0] // 2,
        model.simulation.XY_grid[0].shape[0] // 2
    ]
    im = ax.imshow(
        side_view_intensity.T,
        aspect='auto',
        extent=extent,
        origin='lower',
        cmap='hot'
    )
    ax.set_title('Side View of Beam Propagation')
    ax.set_xlabel('z distance (m)')
    ax.set_ylabel('y position (pixels)')
    fig.colorbar(im, ax=ax, label='Normalized Intensity')

    return fig

def visualize_weights_map(model):
    import matplotlib.pyplot as plt

    weights_map = model.weights_map.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(weights_map, cmap='viridis')
    plt.colorbar(label='Weight')
    plt.title('Target-based Weights Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('target_based_weights_map.png')
    plt.show()
    plt.close()

# In the optimize_phase_mask function, after initializing the model:

if __name__ == "__main__":

    data_dir = "/data/arouxel"  # You can change this to any directory you want
    target_modes = [1,2,3, 4, 5, 6, 7]  # Add or remove modes as needed
    num_runs_per_mode = 5
    num_epochs = 8200  # Set the number of epochs here
    run_name = "Phase_masks_1D_with_lens"  # Optional: provide a custom run name

    run_multiple_tests(data_dir, target_modes, num_runs_per_mode, num_epochs, run_name)


