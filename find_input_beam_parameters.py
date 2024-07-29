import torch
import torch.optim as optim
import h5py
from helpers import load_yaml_config, resample_intensity
from simulation import Simulation
from sources import Source
from cost_functions import *

# Load configuration files
config_source = load_yaml_config("./configs/source.yml")
config_slm = load_yaml_config("./configs/SLM.yml")
config_simulation = load_yaml_config("./configs/simulation.yml")

# Initialize simulation and source
simulation = Simulation(config_dict=config_simulation)
source = Source(config_dict=config_source, XY_grid=simulation.XY_grid)

# Load target intensity from file
with h5py.File("input_beam_image.hdf5", "r") as f:
    target_intensity = f["intensity"][:]
    XY_detector_grid = f["XY_grid"][:]

# Resample target intensity to match simulation grid
target_intensity = resample_intensity(target_intensity, XY_detector_grid, simulation.XY_grid)
target_intensity = torch.tensor(target_intensity, dtype=torch.float32)

# Set requires_grad to True for parameters to be optimized
source.amplitudes.requires_grad = True
source.means.requires_grad = True
source.sigmas.requires_grad = True

# Define optimizer
optimizer = optim.Adam([source.amplitudes, source.means, source.sigmas], lr=0.01)

# Define loss function (e.g., Mean Squared Error)
loss_fn = torch.nn.MSELoss()

# Training loop
num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Generate the current intensity
    amplitude_input_field = source.gaussian_mixture()
    current_intensity = amplitude_input_field ** 2

    # Compute loss
    loss = loss_fn(current_intensity, target_intensity)

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save the optimized parameters
torch.save(source.state_dict(), "optimized_source.pth")

print("Optimization complete.")
