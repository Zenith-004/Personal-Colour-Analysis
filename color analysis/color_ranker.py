# Color Ranker
# This script takes a target color and a list of other colors,
# then ranks them based on their perceptual closeness to the target.

# Installation:
# This script requires scikit-image.
# pip install scikit-image numpy

import numpy as np
from skimage.color import rgb2lab, deltaE_cie76
import re

# --- Helper Functions ---

def hex_to_rgb(hex_color):
    """Converts a HEX color string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def is_valid_hex(hex_code):
    """Validates if a string is a proper hex code."""
    return re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', hex_code)

def rank_colors(target_hex, color_list_hex):
    """
    Ranks a list of hex colors based on their distance to a target hex color.
    Returns a sorted list of tuples: (hex_color, distance).
    """
    # Convert target hex to RGB, then to LAB color space
    # The rgb2lab function expects a 3D array, so we need to reshape
    target_rgb = np.array(hex_to_rgb(target_hex), dtype=np.uint8)
    target_lab = rgb2lab(target_rgb.reshape(1, 1, 3))

    color_distances = []

    for hex_code in color_list_hex:
        # Convert list color to RGB and then to LAB
        current_rgb = np.array(hex_to_rgb(hex_code), dtype=np.uint8)
        current_lab = rgb2lab(current_rgb.reshape(1, 1, 3))

        # Calculate the perceptual distance (Delta E)
        distance = deltaE_cie76(target_lab, current_lab)
        color_distances.append((hex_code, float(distance)))

    # Sort the list by distance (the second element in the tuple), ascending
    color_distances.sort(key=lambda x: x[1])

    return color_distances


# --- Command-Line Interface ---
if __name__ == "__main__":
    print("--- Color Ranker ---")
    
    # Get target color
    target_color = ""
    while not is_valid_hex(target_color):
        target_color = input("Enter the target HEX color (e.g., #ff5733): ")
        if not is_valid_hex(target_color):
            print("Invalid HEX code format. Please try again.")

    # Get list of colors to rank
    color_list_input = input("Enter a list of HEX colors to rank, separated by commas: \n")
    
    # Clean and validate the list
    raw_colors = [color.strip() for color in color_list_input.split(',')]
    valid_colors = [color for color in raw_colors if is_valid_hex(color)]
    
    if not valid_colors:
        print("\nError: No valid HEX colors were provided in the list.")
    else:
        # Rank the colors
        ranked_list = rank_colors(target_color, valid_colors)

        # Display results
        print("\n" + "="*40)
        print(f"Colors ranked by closeness to {target_color}:")
        print("(Lower score means a closer match)")
        print("="*40 + "\n")

        for i, (color, distance) in enumerate(ranked_list):
            print(f"{i+1}. {color} (Score: {distance:.2f})")
