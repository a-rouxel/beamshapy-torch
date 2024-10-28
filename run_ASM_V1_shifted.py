import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from units import *
from helpers import load_yaml_config
from simulation import Simulation
from sources import Source
from slm import SLM
from ft_lens import ASMPropagation, propagate_and_collect_side_view
from FieldGeneration import generate_target_profiles, generate_target_profiles_shifted
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
        self.config_simulation = load_yaml_config("./configs/simulation_ASM.yml")

        # Initialize components
        self.simulation = Simulation(config_dict=self.config_simulation)
        self.source = Source(config_dict=self.config_source, XY_grid=self.simulation.XY_grid)
        self.asm = ASMPropagation(
            self.simulation.delta_x_in,
            self.source.wavelength,
            37 * mm,
            self.simulation.XY_grid[0].shape
        )

        # Set device
        self.device = torch.device(device)

        # Prepare target field
        self.target_mode = target_mode_nb
        if self.target_mode %2 == 0:
            self.mode_parity = "even"
        else:
            self.mode_parity = "odd"
        self.with_minimize_losses = with_minimize_losses
        self.list_target_files = generate_target_profiles_shifted(
            yaml_file="./configs/target_profile.yml",
            XY_grid=self.simulation.XY_grid,
            list_modes_nb=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            position_shift=torch.tensor([1.5,0])*mm
        )
        # Move target fields to device
        self.list_target_files = [field.to(self.device) for field in self.list_target_files]
        self.target_field = self.list_target_files[self.target_mode]

        target_field_amplitude = torch.abs(self.target_field)
        plt.imshow(target_field_amplitude.detach().cpu().numpy())
        plt.savefig("target_field_amplitude.png")
        plt.close()


        self.slm = SLM(config_dict=self.config_slm, XY_grid=self.simulation.XY_grid)

        # Replace the hypergaussian_weights call with the new method
        self.weights_map = self.create_target_based_weights(
            self.target_field,
            sigma=2  # Adjust this value to control the Gaussian blur
        ).to(self.device)

    def forward(self, source_field,epoch):
        # Apply the SLM phase modulation
        modulated_field = self.slm.apply_phase_modulation_sigmoid(source_field,steepness=2 + epoch/200,num_terms=1, spacing=1,mode_parity=self.mode_parity)

        # Propagate the field
        out_field = self.asm(modulated_field)

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
        energy_out = torch.sum(torch.abs(out_field) ** 2)
        energy_out = energy_out + 1e-10  # Avoid divide by zero

        # Apply the weight map to the fields
        weighted_out_field = out_field * self.weights_map

        inside_energy_percentage = torch.sum(torch.abs(weighted_out_field) ** 2) / energy_out

        list_overlaps = []
        for idx, target_field in enumerate(list_target_field):
            energy_target = torch.sum(torch.abs(target_field))
            target_field_norm = target_field * energy_out / energy_target

            weighted_target_field = target_field_norm * self.weights_map
            overlap = calculate_normalized_overlap(out_field, target_field)
            list_overlaps.append(overlap)

        loss1 = -torch.log(list_overlaps[self.target_mode] + 1e-10)

        # Exclude the target mode from the overlaps
        other_overlaps = [overlap for idx, overlap in enumerate(list_overlaps) if idx != self.target_mode]
        max_other_overlap = torch.max(torch.stack(other_overlaps))

        loss2 = -epoch * torch.log(1 - max_other_overlap + 1e-10)

        # Total Variation (TV) regularization for smoothness
        diff_dim0 = torch.diff(self.slm.phase, dim=0) ** 2
        diff_dim1 = torch.diff(self.slm.phase, dim=1) ** 2

        # Pad the differences to match original size
        diff_dim0_padded = nn.functional.pad(diff_dim0, (0, 0, 0, 1), 'constant', 0)
        diff_dim1_padded = nn.functional.pad(diff_dim1, (0, 1, 0, 0), 'constant', 0)

        TV_term = torch.sum(torch.sqrt(diff_dim0_padded + diff_dim1_padded + 1e-2))
        loss3 = TV_term * 5e-6 * (50 / (epoch + 1))
        # loss3 = torch.tensor(0, device=self.device)

            # Symmetry loss
        slm_phase = self.slm.phase
        # slm_phase_flip_x = torch.flip(slm_phase, dims=[0])  # Flip along the X axis
        slm_phase_flip_y = torch.flip(slm_phase, dims=[1])  # Flip along the Y axis
        # symmetry_loss_x = torch.mean((slm_phase - slm_phase_flip_x) ** 2)
        symmetry_loss_y = torch.mean((slm_phase - slm_phase_flip_y) ** 2)
        symmetry_loss = (symmetry_loss_y)

        # Energy inside the target region
        loss4 = (1 - inside_energy_percentage) ** 2

        # Weight for symmetry loss
        symmetry_weight = 1.0  # You can adjust this weight based on importance

        if self.with_minimize_losses:
            loss = loss1 + loss2 + loss3 + loss4 + symmetry_weight * symmetry_loss
        else:
            loss = loss1 + loss2 + loss3 + symmetry_weight * symmetry_loss

        # Collect loss components for logging
        loss_components = {
            'total_loss': loss.item(),
            'loss1': loss1.item(),
            'loss2': loss2.item(),
            'loss3': loss3.item(),
            'loss4': loss4.item(),
            'symmetry_loss': symmetry_loss.item()*symmetry_weight,
            'inside_energy_percentage': inside_energy_percentage.item(),
            'list_overlaps': [overlap.item()*100 for overlap in list_overlaps]
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-3)

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
        loss, loss_components = model.compute_loss(out_field, model.list_target_files, epoch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        global_step = epoch

        # Log scalars with more descriptive names
        writer.add_scalar('Loss/Total', loss.item(), global_step)
        writer.add_scalar('Loss/Target Mode Overlap', loss_components['loss1'], global_step)
        writer.add_scalar('Loss/Other Modes Suppression', loss_components['loss2'], global_step)
        writer.add_scalar('Loss/Phase Smoothness (TV)', loss_components['loss3'], global_step)
        writer.add_scalar('Loss/Energy Confinement', loss_components['loss4'], global_step)
        writer.add_scalar('Loss/Symmetry', loss_components['symmetry_loss'], global_step)
        writer.add_scalar('Metrics/Inside Energy Percentage', loss_components['inside_energy_percentage'], global_step)

        # Log overlaps with mode numbers
        for idx, overlap in enumerate(loss_components['list_overlaps']):
            writer.add_scalar(f'Overlap/Mode {idx}', overlap, global_step)

        # Update progress bar description with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        pbar.update(1)

        if epoch in save_mask_epochs or epoch == num_epochs - 1:  # Add this condition
            # Save the phase mask
            os.makedirs(f'{data_dir}/{run_name}/phase_masks/mode_{target_mode_nb}/run_{run_number}', exist_ok=True)
            np.save(f'{data_dir}/{run_name}/phase_masks/mode_{target_mode_nb}/run_{run_number}/slm_phase_epoch_{epoch}.npy', model.slm.phase.detach().cpu().numpy())

            # # Log images
            # # Phase mask
            # phase_mask = model.slm.phase.detach().cpu().numpy()
            # phase_mask = (phase_mask - phase_mask.min()) / (phase_mask.max() - phase_mask.min() + 1e-8)
            # phase_mask = phase_mask[np.newaxis, :, :]  # Add channel dimension
            # writer.add_image('SLM/phase_mask', phase_mask, global_step)

            # # Output field intensity
            # output_intensity = normalize_image(out_field)
            # output_intensity = output_intensity[np.newaxis, :, :]
            # writer.add_image('Field/output_intensity', output_intensity, global_step)

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

    # Create a progress bar for the overall process
    total_iterations = len(target_modes) * num_runs_per_mode
    overall_pbar = tqdm(total=total_iterations, desc="Overall Progress", position=0, leave=True)

    for mode in target_modes:
        for run in range(num_runs_per_mode):
            # Create a nested progress bar for the current optimization
            optimize_pbar = tqdm(total=num_epochs, desc=f"Mode {mode}, Run {run+1}", position=1, leave=False)
            
            # Pass the progress bar to optimize_phase_mask
            overlaps, inside_energy = optimize_phase_mask(target_mode_nb=mode, run_name=run_name, run_number=run + 1, data_dir=data_dir, num_epochs=num_epochs, pbar=optimize_pbar)
            results.append([mode, run + 1] + overlaps + [inside_energy])

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

    # Save results to CSV
    csv_filename = f"{run_dir}/results.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(results)
    
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
    target_modes = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Add or remove modes as needed
    num_runs_per_mode = 5
    num_epochs = 8200  # Set the number of epochs here
    run_name = "Phase_masks_lensless_37mm_shifted"  # Optional: provide a custom run name

    run_multiple_tests(data_dir, target_modes, num_runs_per_mode, num_epochs, run_name)
