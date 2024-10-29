import klayout.db as db
import numpy as np


dir_path_masks = "/home/arouxel/Documents/POSTDOC/beamshapy-pytorch/comparisons_results_optim_on_selectivity_3um_with_optims/"
mask_type = "optim_masks_lensless"
mask = np.load(dir_path_masks + mask_type + "/full_mask_mode_0.npy")

# Normalize mask to range [0, 1]
mask = (mask - mask.min()) / (mask.max() - mask.min())


# Input parameters
pixel_size = 3.0  # Size of the pixel in micrometers

# Convert mask to binary (0 or 1)
binary_mask = (mask > 0.5).astype(int)



# Create a new layout and a top cell
layout = db.Layout()
top_cell = layout.create_cell("TOP")

# Define the layer
layer_index = layout.insert_layer(db.LayerInfo(1, 0))  # Layer 1, datatype 0

# Loop through the binary mask and create squares for each '1'
for row_idx, row in enumerate(binary_mask):
    for col_idx, value in enumerate(row):
        if value == 1:  # If the cell value is 1, create a pixel
            # Calculate the square's coordinates
            x_min = col_idx * pixel_size
            y_min = row_idx * pixel_size
            x_max = x_min + pixel_size
            y_max = y_min + pixel_size

            # Create a rectangle for the pixel
            pixel = db.Box(x_min, y_min, x_max, y_max)
            top_cell.shapes(layer_index).insert(pixel)

# Save the layout to a file
output_file = "./klayout_masks/generated_mask.gds"
layout.write(output_file)

print(f"Layout saved to {output_file}")
