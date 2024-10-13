import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from simulation import Simulation
from sources import Source
from ft_lens import ASMPropagation, propagate_and_collect_side_view
from units import *
from helpers import load_yaml_config


## Binary Fresnel Lens Mask
# Parameters
width, height = 500, 500  # Mask dimensions
focal_length = 5000  # Focal length in pixels
wavelength = 1  # Arbitrary units

# Create coordinate grid
x = np.linspace(-width//2, width//2, width)
y = np.linspace(-height//2, height//2, height)
X, Y = np.meshgrid(x, y)
R_squared = X**2 + Y**2

# Calculate phase pattern
phase = (np.pi / wavelength / focal_length) * R_squared
binary_mask = np.mod(phase, 2 * np.pi) < np.pi  # Binary conversion

# Convert to 0 and 1
binary_mask = binary_mask.astype(np.uint8) * 255

# New cropping function
def crop_circular(image, radius):
    center_x, center_y = width // 2, height // 2
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    circular_mask = dist_from_center <= radius
    cropped_image = image.copy()
    cropped_image[~circular_mask] = 0
    return cropped_image

# Apply circular crop
crop_radius = 195 # Adjust this value to change the crop size
cropped_binary_mask = crop_circular(binary_mask, crop_radius)

# Oversample the cropped binary mask by a factor of 4
oversampling_factor = 4
oversampled_height, oversampled_width = cropped_binary_mask.shape[0] * oversampling_factor, cropped_binary_mask.shape[1] * oversampling_factor

oversampled_mask = np.zeros((oversampled_height, oversampled_width), dtype=np.uint8)

for i in range(oversampled_height):
    for j in range(oversampled_width):
        oversampled_mask[i, j] = cropped_binary_mask[i // oversampling_factor, j // oversampling_factor]

# Display the oversampled mask
plt.figure(figsize=(10, 10))
plt.imshow(oversampled_mask, cmap='gray')
plt.title(f'Oversampled Cropped Binary Fresnel Lens (4x)')
plt.axis('off')
plt.show()

# Save the oversampled mask
Image.fromarray(oversampled_mask).save('oversampled_cropped_binary_fresnel_lens.png')

print(f"Oversampled mask shape: {oversampled_mask.shape}")



mask_dithered = np.load("mask_dithered_1.npy")

# After saving the oversampled mask and before loading the dithered mask

# Perform XOR operation
# First, ensure both masks have the same dimensions
mask_dithered_resized = Image.fromarray(mask_dithered).resize((oversampled_width, oversampled_height), Image.NEAREST)
mask_dithered_resized = np.array(mask_dithered_resized)

# XOR operation
xor_result = np.logical_xor(oversampled_mask > 0, mask_dithered_resized > 0).astype(np.uint8) * 255

width = 2000
height = 2000
crop_radius = 195*4 # Adjust this value to change the crop size
xor_result = crop_circular(xor_result, crop_radius)


# Display the XOR result
plt.figure(figsize=(10, 10))
plt.imshow(cropped_binary_mask, cmap='gray')
plt.title('XOR of Oversampled Fresnel Lens and Dithered Mask')
plt.axis('off')
plt.show()

# Optionally, save the XOR result
Image.fromarray(xor_result).save('xor_result_fresnel_and_dithered.png')

# Display the original mask
plt.figure(figsize=(10, 10))
plt.imshow(mask_dithered, cmap='gray')
plt.title('Dithered_mask')
plt.axis('off')
plt.show()




