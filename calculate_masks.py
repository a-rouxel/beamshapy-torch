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
from helpers import generate_target_amplitude, design_mask, generate_target_mask, wrap_phase
import cProfile
import pstats


def main():
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

    width = 52*um
    height = 52*um

    sinus_period = 2*width / (i+1)
    if i % 2 == 0:
        phase_off = np.pi/2
    else:
        phase_off = 0

    target_amplitude = generate_target_amplitude(simulation.XY_grid, ft_lens.XY_output_grid, source.wavelength,ft_lens.focal_length,
                                                 amplitude_type="Rectangle",width=width,height=height)
    target_amplitude *= generate_target_amplitude(simulation.XY_grid, ft_lens.XY_output_grid, source.wavelength,ft_lens.focal_length,
                                                    amplitude_type="Sinus",period=sinus_period,phase_offset=phase_off)



    inverse_fourier_transform = ft_lens(target_amplitude, pad=False, flag_ifft=True)

    wedge_mask = design_mask(simulation.XY_grid,"Wedge",source.wavelength,ft_lens.focal_length,angle=0,position=0.12*mm)
    phase_inversion_mask = generate_target_mask(inverse_fourier_transform,mask_type="phase target field")
    amplitude_modulation_mask, uncorrected_amplitude_mask,_ = generate_target_mask(inverse_fourier_transform,mask_type="modulation amplitude",input_field=source.field.field)


    mask_1 =  wrap_phase(phase_inversion_mask + wedge_mask)*amplitude_modulation_mask


    slm = SLM(config_dict=config_slm, XY_grid=simulation.XY_grid, initial_phase=mask_1)
    #
    modulated_field = slm.apply_phase_modulation(source.field.field, mapping=False)
    out_field = ft_lens(modulated_field, pad=False, flag_ifft=False)
    #
    # plt.imshow(torch.abs(out_field).detach().numpy())
    # plt.colorbar()
    # plt.title("out_field")
    # plt.show()
    #


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)  # Print the top 10 functions by time spent