from helpers import load_yaml_config
from simulation import Simulation
from sources import Source
from slm import SLM
from ft_lens import FT_Lens
from units import *
from FieldGeneration import generate_target_profiles
from cost_functions import calculate_normalized_overlap, quantize_phase
import matplotlib.pyplot as plt
import torch
import numpy as np

def hypergaussian_weights(XY_grid, x_center, y_center, sigma_x, sigma_y, order):
    # Calculate distances from center
    x_grid = XY_grid[0]  # X coordinates of the grid
    y_grid = XY_grid[1]  # Y coordinates of the grid

    x_dist = (x_grid - x_center) ** 2 / (2 * sigma_x ** 2)
    y_dist = (y_grid - y_center) ** 2 / (2 * sigma_y ** 2)

    # Calculate hypergaussian weights
    weights_x = torch.exp(-x_dist ** order)
    weights_y = torch.exp(-y_dist ** order)

    # Produce a square-shaped weight map by multiplying independent x and y weights
    weight_map = weights_x * weights_y
    weight_map /= weight_map.sum()  # Normalize the weights to sum to 1
    return weight_map

def compute_loss_no_weights(out_field, list_target_field):

    list_overlaps = []

    for idx, target_field in enumerate(list_target_field):

        overlap = calculate_normalized_overlap(out_field, target_field)


        list_overlaps.append(overlap)


    return list_overlaps
def compute_loss(out_field, list_target_field,weights_map):
    energy_out = torch.sum(torch.sum(torch.abs(out_field)))

    weights_map = weights_map / torch.max(weights_map)

    weighted_out_field = (out_field * weights_map)
    list_overlaps = []

    for idx, target_field in enumerate(list_target_field):
        energy_target = torch.sum(torch.sum(torch.abs(target_field)))
        target_field_norm = target_field * energy_out / energy_target

        weighted_target_field = (target_field_norm * weights_map)

        # plt.imshow(torch.abs(weighted_out_field).detach().numpy())
        # plt.show()
        overlap = calculate_normalized_overlap(weighted_out_field, weighted_target_field)


        power_weighted_out = torch.sum(torch.abs(weighted_out_field)**2).detach().numpy()
        power_out  = torch.sum(torch.abs(out_field)**2).detach().numpy()

        # print(power_weighted_out)
        # print(power_out)
        # print(power_weighted_out/power_out)

        list_overlaps.append(overlap)


    return list_overlaps


config_source = load_yaml_config("./configs/source.yml")
config_target = load_yaml_config("./configs/target_profile.yml")
config_slm = load_yaml_config("./configs/SLM.yml")
config_simulation = load_yaml_config("./configs/simulation.yml")


i = 2

if i%2 == 0:
    mode_parity = "even"
else:
    mode_parity = "odd"

simulation = Simulation(config_dict=config_simulation)

source = Source(config_dict= config_source,
                XY_grid= simulation.XY_grid)


ft_lens = FT_Lens(simulation.delta_x_in, simulation.XY_grid,source.wavelength)


phase = np.load(f"/home/arouxel/Documents/POSTDOC/beamshapy-pytorch/lightning_logs/best_slm_phase_{i}_0.npy")
phase = torch.tensor(phase).float()

# Quantization
quantized_phase = quantize_phase(phase, 2,mode_parity=mode_parity)

# Display phases
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(phase)
# ax[0].set_title("Original Phase")
# ax[1].imshow(quantized_phase)
# ax[1].set_title("Quantized Phase")
# plt.show()
#
# Generate target profiles
list_target_fields = generate_target_profiles(yaml_file="./configs/target_profile.yml", XY_grid=ft_lens.XY_output_grid, list_modes_nb=[0,1,2,3,4,5,6,7,8])
target_field = list_target_fields[i]

# Simulate field modulation for both original and quantized phases
for phase_type, phase_data in [("Original", phase), ("Quantized", quantized_phase)]:

    # path_partage_photo = "/net/cubitus/projects/Partage_PHOTO/Projets_en_cours/ANR RESON/Echantillons/XB-FBINA-22/optimized_masks/"

    # np.save(f"{path_partage_photo}best_slm_phase_{i}_{try_nb}.npy", phase_data.cpu().detach().numpy())
    plt.imshow(phase_data.cpu().detach().numpy())
    plt.show()

    slm = SLM(config_dict=config_slm, XY_grid=simulation.XY_grid, initial_phase=phase_data)
    modulated_field = slm.apply_phase_modulation(source.field.field, mapping=False)
    out_field = ft_lens(modulated_field, pad=False, flag_ifft=False)
    # #
    # # Display the resulting fields
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(torch.abs(out_field).cpu().detach().numpy())
    ax[0].set_title(f"Output Field ({phase_type})")
    ax[1].imshow(torch.abs(target_field).cpu().detach().numpy())
    ax[1].set_title("Target Field")
    plt.show()
    #
    # # Compute overlap values
    weights_map = hypergaussian_weights(XY_grid=ft_lens.XY_output_grid, x_center=0, y_center=0, sigma_x=19 * um, sigma_y=19 * um, order=12)



    list_overlap = compute_loss(out_field, list_target_fields, weights_map)


    print(f"Overlap Values ({phase_type}): {list_overlap}")