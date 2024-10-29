import os
import klayout.db as db
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
import csv

# Constants
PIXEL_SIZE = 3.0  # Size of the pixel in micrometers
LAYOUT_DBU = 0.001  # Database unit in micrometers (1 nm)
THRESHOLD = 0.5  # Threshold for binary mask conversion

# Input and output paths
DIR_PATH_MASKS = "/home/arouxel/Documents/POSTDOC/beamshapy-pytorch/comparisons_results_optim_on_selectivity_3um_with_optims/"
OUTPUT_DIR = "./klayout_masks/"

MASK_TYPES = [
    "phase_inversion_dithering_masks",
    "phase_inversion_dithering_fresnel_masks",
    "optim_masks_with_lens",
    "optim_masks_lensless"
]
MODE_NBS = range(9)  # 0 to 8

def group_pixels(binary_mask):
    """Group adjacent pixels into labeled groups."""
    structure = np.ones((3, 3), dtype=int)
    labeled_mask, num_features = label(binary_mask, structure=structure)
    return labeled_mask, num_features

def create_polygon(layout, top_cell, layer_index, group, pixel_size):
    """Create polygons for a group of connected pixels."""
    ys, xs = np.where(group)
    if len(xs) == 0:
        return
    region = db.Region()
    scale = pixel_size / layout.dbu  # Convert micrometers to database units
    for y, x in zip(ys, xs):
        x1, y1 = x * scale, y * scale
        x2, y2 = (x + 1) * scale, (y + 1) * scale
        box = db.Box(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        region.insert(box)
    region.merge()
    top_cell.shapes(layer_index).insert(region)

def count_polygons_and_vertices(top_cell, layer_index):
    polygon_count = 0
    vertex_count = 0
    shapes = top_cell.shapes(layer_index)
    for shape in shapes.each():
        if shape.is_polygon():
            polygon = shape.polygon
            polygon_count += 1
            vertex_count += polygon.num_points()
    return polygon_count, vertex_count

def process_mask(mask_path, output_path):
    """Process a single mask and generate the corresponding GDS file."""
    # Load and normalize mask
    mask = np.load(mask_path)
    mask = (mask - mask.min()) / (mask.max() - mask.min())

    # Convert mask to binary
    binary_mask = (mask > THRESHOLD).astype(int)

    # Group adjacent pixels
    labeled_mask, num_groups = group_pixels(binary_mask)

    print(f"Number of groups: {num_groups}")

    # Create a new layout and a top cell
    layout = db.Layout()
    layout.dbu = LAYOUT_DBU
    top_cell = layout.create_cell("TOP")

    # Define the layer
    layer_index = layout.insert_layer(db.LayerInfo(1, 0))  # Layer 1, datatype 0

    # Create polygons for each group
    for group_id in range(1, num_groups + 1):
        print(f"Processing group {group_id}/{num_groups}")
        group = (labeled_mask == group_id)
        create_polygon(layout, top_cell, layer_index, group, PIXEL_SIZE)

    # Save the layout to a file
    layout.write(output_path)
    print(f"Optimized layout saved to {output_path}")

    # Count polygons and vertices
    polygon_count, vertex_count = count_polygons_and_vertices(top_cell, layer_index)
    print(f"Number of polygons: {polygon_count}")
    print(f"Number of vertices: {vertex_count}")

    return polygon_count, vertex_count

import csv

def main():
    """Main function to process all mask types and modes."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    info_file_path = os.path.join(OUTPUT_DIR, "masks_info.csv")
    with open(info_file_path, 'w', newline='') as info_file:
        csv_writer = csv.writer(info_file)
        csv_writer.writerow(["Mask Type", "Mode Number", "Number of Polygons", "Number of Vertices"])
        for mask_type in MASK_TYPES:
            for mode_nb in MODE_NBS:
                mask_path = os.path.join(DIR_PATH_MASKS, mask_type, f"full_mask_mode_{mode_nb}.npy")
                output_path = os.path.join(OUTPUT_DIR, f"{mask_type}_mode_{mode_nb}.gds")
                
                print(f"Processing {mask_type}, mode {mode_nb}")
                polygon_count, vertex_count = process_mask(mask_path, output_path)
                # Write info to CSV file
                csv_writer.writerow([mask_type, mode_nb, polygon_count, vertex_count])

if __name__ == "__main__":
    main()


