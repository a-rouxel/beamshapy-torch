from helpers import load_yaml_config
from simulation import Simulation
from sources import Source
from slm import SLM
from ft_lens import FT_Lens
from units import *
from FieldGeneration import generate_target_profiles
from cost_functions import calculate_normalized_overlap


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


def compute_loss(out_field, list_target_field,weights_map):
    energy_out = torch.sum(torch.sum(torch.abs(out_field)))
    weighted_out_field = (out_field * weights_map)
    list_overlaps = []

    for idx, target_field in enumerate(list_target_field):
        energy_target = torch.sum(torch.sum(torch.abs(target_field)))
        target_field_norm = target_field * energy_out / energy_target

        weighted_target_field = (target_field_norm * weights_map)

        overlap = calculate_normalized_overlap(weighted_out_field, weighted_target_field)

        list_overlaps.append(overlap)


    return list_overlaps


config_source = load_yaml_config("./configs/source.yml")
config_target = load_yaml_config("./configs/target_profile.yml")
config_slm = load_yaml_config("./configs/SLM.yml")
config_simulation = load_yaml_config("./configs/simulation.yml")

for i in range(6):


    simulation = Simulation(config_dict=config_simulation)

    source = Source(config_dict= config_source,
                    XY_grid= simulation.XY_grid)


    ft_lens = FT_Lens(simulation.delta_x_in, simulation.XY_grid,source.wavelength)


    list_target_fields = generate_target_profiles(yaml_file="./configs/target_profile.yml", XY_grid= ft_lens.XY_output_grid,list_modes_nb=[0,1,2,3,4,5,6])
    inverse_fourier_field = ft_lens(list_target_fields[i], pad=False, flag_ifft=True)


    import matplotlib.pyplot as plt
    import torch

    # plt.plot(torch.imag(inverse_fourier_field[:,1250]).float())
    # plt.show()
    phase = torch.angle(inverse_fourier_field)

    # if abs(phase) <0.01 replace with 0
    phase[torch.abs(phase) < 0.01] = 0
    phase[torch.abs(phase) > 3.1] = torch.pi

    #
    # plt.imshow(torch.abs(inverse_fourier_field))
    # plt.show()
    # plt.imshow(phase)
    # plt.savefig(f"phase_{i}.png")
    # plt.show()

    target_field = list_target_fields[i]

    slm = SLM(config_dict= config_slm,
                XY_grid= simulation.XY_grid,initial_phase=phase)




    modulated_field = slm.apply_phase_modulation(source.field.field,mapping=False)

    phase_modulated_field = torch.angle(modulated_field)



    out_field = ft_lens(modulated_field, pad=False, flag_ifft=False)


    weights_map = hypergaussian_weights(XY_grid=ft_lens.XY_output_grid, x_center=0, y_center=0,
                                                  sigma_x=19 * um, sigma_y=19 * um, order=12)

    list_overlap = compute_loss(out_field, list_target_fields,weights_map)

    print(list_overlap)