from helpers import load_yaml_config
from simulation import Simulation
from sources import Source
from slm import SLM
from ft_lens import FT_Lens
from electric_field import ElectricField
from units import *
import torch

import matplotlib.pyplot as plt

from FieldGeneration import generate_profile

config_source = load_yaml_config("./configs/source.yml")
config_slm = load_yaml_config("./configs/SLM.yml")
config_simulation = load_yaml_config("./configs/simulation.yml")

simulation = Simulation(config_dict=config_simulation)

# load the target field

# radius = 2*mm
# parabola_coef = 10**6
# hyper_gauss_order = 12

# target_field = generate_profile("Fresnel Lens", simulation.XY_grid,radius, parabola_coef)


width = 520*um
height = 520*um
sinus_period = 500*um / 1.5 

target_field = generate_profile("Rectangle",simulation.XY_grid,width=width,height=height)
target_field *= generate_profile("Sinus",simulation.XY_grid,period=sinus_period,phase_offset=torch.pi/2)
target_field = ElectricField(torch.abs(target_field), torch.angle(target_field), simulation.XY_grid)

target_field.show_field()


source = Source(config_dict= config_source, 
                XY_grid= simulation.XY_grid)
    
slm = SLM(config_dict= config_slm,
            XY_grid= simulation.XY_grid)

ft_lens = FT_Lens(simulation.delta_x_in, simulation.XY_grid,source.wavelength)

out_field = ft_lens(source.field, pad=False, flag_ifft=False)

# out_field.show_field()