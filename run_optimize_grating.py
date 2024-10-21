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
from FieldGeneration import generate_target_profiles, generate_multiple_gaussian
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
    def __init__(self, device="cpu"):
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

        self.target_field = generate_multiple_gaussian(number_of_gaussians_x=3,
                                                            number_of_gaussians_y=3,
                                                            spacing_x=1e-3,
                                                            spacing_y=1e-3,
                                                            sigma=10*1e-6,
                                                            XY_grid=self.simulation.XY_grid
                                                     ).field.to(self.device)
        
        # plt.imshow(torch.abs(self.target_field).detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()

        self.slm = SLM(config_dict=self.config_slm, XY_grid=self.simulation.XY_grid)


    def forward(self, source_field,epoch):
        # Apply the SLM phase modulation
        modulated_field = self.slm.apply_phase_modulation_sigmoid(source_field,steepness=2 + epoch/200,num_terms=1, spacing=1,mode_parity="odd")

        # Propagate the field
        out_field = self.ft_lens(modulated_field, pad=False, flag_ifft=False)

        return out_field



    def compute_loss(self, out_field, target_field, epoch):

        energy_out = torch.sum(torch.abs(out_field) ** 2)
        energy_out = energy_out + 1e-10  # Avoid divide by zero

        # Apply the weight map to the fields

        inside_energy_percentage = torch.sum(torch.abs(out_field) ** 2) / energy_out

            # energy_target = torch.sum(torch.abs(target_field))
            # target_field_norm = target_field * energy_out / energy_target

            # weighted_target_field = target_field_norm * self.weights_map
        overlap = calculate_normalized_overlap(out_field, target_field)

        loss1 = -torch.log(overlap + 1e-10)


        # Total Variation (TV) regularization for smoothness
        diff_dim0 = torch.diff(self.slm.phase, dim=0) ** 2
        diff_dim1 = torch.diff(self.slm.phase, dim=1) ** 2

        # Pad the differences to match original size
        diff_dim0_padded = nn.functional.pad(diff_dim0, (0, 0, 0, 1), 'constant', 0)
        diff_dim1_padded = nn.functional.pad(diff_dim1, (0, 1, 0, 0), 'constant', 0)

        TV_term = torch.sum(torch.sqrt(diff_dim0_padded + diff_dim1_padded + 1e-2))
        loss3 = TV_term * 5e-6 * (50 / (epoch + 1))
        # loss3 = torch.tensor(0, device=self.device)


        # Energy inside the target region
        loss4 = 0.2 * (1 - inside_energy_percentage) ** 2



        loss = loss1  + loss3 + loss4 


        # Collect loss components for logging
        loss_components = {
            'total_loss': loss.item(),
            'loss1': loss1.item(),
            'loss3': loss3.item(),
            'loss4': loss4.item(),
            'inside_energy_percentage': inside_energy_percentage.item()
        }

        return loss, loss_components

def normalize_image(tensor):
    tensor = tensor.detach().cpu().numpy()
    tensor = np.abs(tensor)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    return tensor

def optimize_phase_mask( run_name, run_number, data_dir, num_epochs, pbar):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the TensorBoard writer with a unique run name
    writer = SummaryWriter(log_dir=f'{data_dir}/{run_name}/logs/run_{run_number}')

    # Initialize the model
    model = OpticalSystem(device=device)

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
        loss, loss_components = model.compute_loss(out_field, model.target_field, epoch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        global_step = epoch

        # Log scalars with more descriptive names
        writer.add_scalar('Loss/Total', loss.item(), global_step)
        writer.add_scalar('Loss/Target Mode Overlap', loss_components['loss1'], global_step)
        writer.add_scalar('Loss/Phase Smoothness (TV)', loss_components['loss3'], global_step)
        writer.add_scalar('Loss/Energy Confinement', loss_components['loss4'], global_step)
        writer.add_scalar('Metrics/Inside Energy Percentage', loss_components['inside_energy_percentage'], global_step)

        # Update progress bar description with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        pbar.update(1)

        if epoch in save_mask_epochs:
            # Save the phase mask
            os.makedirs(f'{data_dir}/{run_name}/phase_masks/run_{run_number}', exist_ok=True)
            np.save(f'{data_dir}/{run_name}/phase_masks/run_{run_number}/slm_phase_epoch_{epoch}.npy', model.slm.phase.detach().cpu().numpy())

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

            # Target field intensity
            target_intensity = normalize_image(model.target_field)
            target_intensity = target_intensity[np.newaxis, :, :]
            writer.add_image('Field/target_intensity', target_intensity, global_step)

            # # Optional: Log side-view propagation figure
            # fig = plot_side_view(model, source_field)
            # writer.add_figure('Propagation/side_view', fig, global_step)
            # plt.close(fig)

    # Return the final results
    final_overlaps = loss_components['list_overlaps']
    final_inside_energy = loss_components['inside_energy_percentage']*100
    return final_overlaps, final_inside_energy

def run_multiple_tests(data_dir, num_runs_per_mode, num_epochs, run_name=None):
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



    for run in range(num_runs_per_mode):
        # Create a nested progress bar for the current optimization
        optimize_pbar = tqdm(total=num_epochs, desc=f"Mode 0 Run {run+1}", position=1, leave=False)
        
        # Pass the progress bar to optimize_phase_mask
        overlaps, inside_energy = optimize_phase_mask(run_name=run_name, run_number=run + 1, data_dir=data_dir, num_epochs=num_epochs, pbar=optimize_pbar)
        result = ['init', run + 1] + overlaps + [inside_energy]
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
        
    
    print(f"\nResults saved to {csv_filename}")



# In the optimize_phase_mask function, after initializing the model:

if __name__ == "__main__":

    data_dir = "/data/arouxel"  # You can change this to any directory you want
    num_runs_per_mode = 3
    num_epochs = 8200  # Set the number of epochs here
    run_name = "grating_optimization"  # Optional: provide a custom run name

    run_multiple_tests(data_dir, num_runs_per_mode, num_epochs, run_name)
