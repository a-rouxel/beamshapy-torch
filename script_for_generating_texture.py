import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


## Binary Fresnel Lens Mask
# Parameters
width, height = 512, 512  # Mask dimensions
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

# Display the cropped image
plt.imshow(cropped_binary_mask, cmap='gray')
plt.title(f'Cropped Binary Fresnel Lens (radius: {crop_radius} pixels)')
plt.show()

# Save the cropped mask
image = Image.fromarray(cropped_binary_mask)
image.save('cropped_binary_fresnel_lens.png')


## Mask with Varying Density of Studs Increasing with Y
import numpy as np
from PIL import Image

# Floyd-Steinberg dithering algorithm from compare_approaches.py
def floyd_steinberg(image):
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[y, x]
            new = np.round(old)
            image[y, x] = new
            error = old - new
            if x + 1 < w:
                image[y, x + 1] += error * 0.4375  # right, 7 / 16
            if (y + 1 < h) and (x + 1 < w):
                image[y + 1, x + 1] += error * 0.0625  # right, down, 1 / 16
            if y + 1 < h:
                image[y + 1, x] += error * 0.3125  # down, 5 / 16
            if (x - 1 >= 0) and (y + 1 < h):
                image[y + 1, x - 1] += error * 0.1875  # left, down, 3 / 16
    return image

# Parameters
width, height = 16, 16  # Original mask dimensions
density_min = 0
density_max = 1

# Create a linear density gradient along Y-axis
density_values = np.linspace(density_min, density_max, height)
density_map = np.tile(density_values, (width, 1)).T  # Transpose to match dimensions

# Apply Floyd-Steinberg dithering
dithered_mask = floyd_steinberg(density_map)

# Threshold the dithered image to create a binary mask
binary_mask = (dithered_mask > 0.5).astype(np.uint8) * 255

# Display the original mask
plt.figure(figsize=(10, 10))
plt.imshow(binary_mask, cmap='gray')
plt.title('Varying Density Studs (Floyd-Steinberg Dithering)')
plt.axis('off')
plt.show()

# Oversample by a factor of 50
oversampling_factor = 50
oversampled_mask = np.repeat(np.repeat(binary_mask, oversampling_factor, axis=0), oversampling_factor, axis=1)

# Apply padding after oversampling
padding = 1  # Padding in oversampled pixels
padded_height, padded_width = oversampled_mask.shape
padded_mask = np.zeros((padded_height + 2*padding, padded_width + 2*padding), dtype=np.uint8)
padded_mask[padding:-padding, padding:-padding] = oversampled_mask

# Display the oversampled and padded mask
plt.figure(figsize=(10, 10))
plt.imshow(padded_mask, cmap='gray')
plt.title('Oversampled Varying Density Studs with Padding (50x)')
plt.axis('off')
plt.show()

# Save the oversampled and padded mask
image = Image.fromarray(padded_mask)
image.save('varying_density_studs_floyd_steinberg_oversampled_padded.png')


## 3 - phase inversion

# Function to crop borders
def crop_borders(image, border_size):
    h, w = image.shape
    return image[border_size:h-border_size, border_size:w-border_size]

# Function to add padding
def add_padding(image, padding):
    h, w = image.shape
    padded_image = np.zeros((h + 2*padding, w + 2*padding), dtype=image.dtype)
    padded_image[padding:-padding, padding:-padding] = image
    return padded_image

# Load the phase inversion mask
phase_inversion_mask = np.load("phase_inversion_mask.npy")

# Crop the borders
border_size = 0  # Adjust this value to change the amount of cropping
cropped_mask = crop_borders(phase_inversion_mask, border_size)

# Normalize the cropped mask to range [0, 1]
normalized_mask = (cropped_mask - np.min(cropped_mask)) / (np.max(cropped_mask) - np.min(cropped_mask))

# Scale to range [0, 255], convert to uint8, and apply binary threshold
adapted_mask = (normalized_mask * 255).astype(np.uint8)
binary_mask = np.where(adapted_mask >= 128, 255, 0).astype(np.uint8)

# Add padding
padding = 2  # Adjust this value to change the amount of padding
padded_binary_mask = add_padding(binary_mask, padding)

# Display the padded binary mask with a red circle
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(padded_binary_mask, cmap='gray')

# Calculate the radius in pixels (assuming 300 DPI)
dpi = 300
radius_mm = 1.5
radius_pixels = int(radius_mm * dpi)  # Convert mm to pixels

# Add the red circle
center = (padded_binary_mask.shape[1] // 2, padded_binary_mask.shape[0] // 2)
circle = Circle(center, radius_pixels, fill=False, edgecolor='red', linewidth=2)
ax.add_patch(circle)

ax.set_title(f'Padded Binary Phase Inversion Mask with 1.5 mm radius circle\n(borders cropped by {border_size} pixels, padding: {padding} pixel)')
plt.show()

# Save the padded binary mask with the circle
fig.savefig('padded_binary_cropped_phase_inversion_mask_with_circle.png', dpi=300, bbox_inches='tight')

# Save the original padded binary mask without the circle
Image.fromarray(padded_binary_mask).save('padded_binary_cropped_phase_inversion_mask.png')
