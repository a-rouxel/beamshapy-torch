from helpers import load_yaml_config
from simulation import Simulation
from sources import Source
from slm import SLM
from ft_lens import FT_Lens


config_source = load_yaml_config("./configs/source.yml")
config_slm = load_yaml_config("./configs/SLM.yml")
config_simulation = load_yaml_config("./configs/simulation.yml")

simulation = Simulation(config_dict=config_simulation)

source = Source(config_dict= config_source, 
                XY_grid= simulation.XY_grid)
    
slm = SLM(config_dict= config_slm,
            XY_grid= simulation.XY_grid)


ft_lens = FT_Lens(simulation.delta_x_in, simulation.XY_grid,source.wavelength)

out_field = ft_lens(source.field, pad=False, flag_ifft=False)

# out_field.show_field()