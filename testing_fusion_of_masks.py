from simulation import Simulation
from sources import Source
from ft_lens import ASMPropagation, create_binary_fresnel_lens, create_binary_fresnel_lens_new, propagate_and_collect_side_view
from units import *
from helpers import load_yaml_config
import matplotlib.pyplot as plt
import numpy as np
import torch 



config_source = load_yaml_config("./configs/source.yml")
config_target = load_yaml_config("./configs/target_profile.yml")
config_slm = load_yaml_config("./configs/SLM.yml")
config_input_beam = load_yaml_config("./configs/input_beam.yml")
config_optical_system = load_yaml_config("./configs/optical_system.yml")


simulation_asm = Simulation(config_dict=load_yaml_config("./configs/simulation_ASM.yml"))
source_asm = Source(config_dict=config_source, XY_grid=simulation_asm.XY_grid)
asm = ASMPropagation(simulation_asm.delta_x_in, source_asm.wavelength, 20*mm, simulation_asm.XY_grid[0].shape) 


wavelength = 850e-9  # 633 nm (He-Ne laser)
focal_length = 150*mm  # Focal length in meters
radius = 1*mm

# Generate the binary Fresnel lens phase mask

phase_mask, phase_non_quantized = create_binary_fresnel_lens(simulation_asm.XY_grid[0].shape, (simulation_asm.delta_x_in, simulation_asm.delta_x_in), wavelength, focal_length,radius=radius)

np.save("phase_mask.npy", phase_mask.angle().numpy())

plt.imshow(phase_mask.angle())
plt.show()

mask_dithered = np.load("mask_dithered_2.npy")
mask_dithered += np.pi/2

phase_mask_angle = phase_mask.angle() + np.pi

combined_mask = (phase_mask_angle + mask_dithered)
combined_mask_mod = combined_mask % (2*np.pi)

# Create a circular mask with radius 1 mm
center_x, center_y = simulation_asm.XY_grid[0].shape[0] // 2, simulation_asm.XY_grid[1].shape[1] // 2
Y, X = np.ogrid[:simulation_asm.XY_grid[0].shape[0], :simulation_asm.XY_grid[1].shape[1]]
dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
mask = dist_from_center <= (radius / simulation_asm.delta_x_in)

# Apply the mask to combined_mask_mod
combined_mask_mod = np.where(mask, combined_mask_mod, 0)
combined_mask_mod = torch.tensor(combined_mask_mod)

# Convert to complex exponential form
combined_mask_mod = torch.exp(1j * combined_mask_mod)




# fig, ax = plt.subplots(1,4)
# ax[0].imshow(phase_mask_angle)
# ax[1].imshow(mask_dithered)
# ax[2].imshow(combined_mask)
# ax[3].imshow(combined_mask_mod.angle())
# plt.show()

# Modify z_values to have more points for a smoother visualization
z_values = np.linspace(0*mm,15*mm, 100)


u_in_masked = source_asm.field.field * combined_mask_mod
# Generate the side view intensity pattern

# side_view_intensity = propagate_and_collect_side_view(u_in_masked,simulation_asm.delta_x_in, source_asm.wavelength, z_values, simulation_asm.XY_grid[0].shape)


# plt.figure()
# extent = [z_values[0], z_values[-1], -simulation_asm.XY_grid[0].shape[0]//2, simulation_asm.XY_grid[0].shape[0]//2]
# plt.imshow(side_view_intensity.T, aspect='auto', origin='lower', cmap='hot',extent=extent)
# plt.show()

# fig, ax = plt.subplots(1,2)

# ax[0].imshow(torch.abs(u_in_masked)**2)
# ax[1].imshow(u_in_masked.angle())
# plt.show()





u_out = asm(u_in_masked)

height = 60*um

# Create a 2D mask for the square region
mask_inside = (simulation_asm.XY_grid[0] > -height/2) & (simulation_asm.XY_grid[0] < height/2) & \
              (simulation_asm.XY_grid[1] > -height/2) & (simulation_asm.XY_grid[1] < height/2)

# Calculate the total energy and the energy inside the mask
total_energy = torch.sum(torch.abs(u_out)**2)
energy_inside = torch.sum(torch.abs(u_out[mask_inside])**2)

# Calculate the proportion of energy inside the mask
energy_proportion = energy_inside / total_energy

print(f"Proportion of energy inside the {height*1e6:.1f} µm square: {energy_proportion:.2%}")

# Plot the intensity distribution with the mask outline
plt.figure(figsize=(10, 8))
extent = [simulation_asm.XY_grid[0][0,0].item(), simulation_asm.XY_grid[0][-1,0].item(),
          simulation_asm.XY_grid[1][0,0].item(), simulation_asm.XY_grid[1][0,-1].item()]
plt.imshow(torch.abs(u_out)**2, cmap='hot', extent=extent)
plt.colorbar(label='Intensity')

# Add mask outline
mask_outline = plt.Rectangle((-height/2, -height/2), height, height, 
                             fill=False, edgecolor='white', linestyle='--')
plt.gca().add_patch(mask_outline)

plt.title(f"Intensity distribution with {height*1e6:.1f} µm square mask")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.show()

