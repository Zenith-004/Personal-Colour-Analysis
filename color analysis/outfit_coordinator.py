# Outfit Color Coordinator (Advanced)
# This script suggests the best items from your wardrobe to build a
# coordinated outfit around a single piece.

# Installation:
# This script requires scikit-image for color science calculations.
# pip install scikit-image numpy

import colorsys
import re
import numpy as np
from skimage.color import rgb2lab, deltaE_cie76

# --- Color Conversion and Validation ---

def hex_to_rgb(hex_color):
    """Converts a HEX color string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    """Converts an (R, G, B) tuple to a HEX color string."""
    r, g, b = int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])
    return f'#{r:02x}{g:02x}{b:02x}'

def is_valid_hex(hex_code):
    """Validates if a string is a proper hex code."""
    return re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', hex_code)

# --- Core Logic ---

def get_ideal_harmony_colors(hex_color):
    """
    Generates a palette of ideal harmony colors (complementary, analogous, etc.)
    and returns them as a single list.
    """
    # Convert hex to RGB, then normalize to 0-1 for colorsys
    r, g, b = [x / 255.0 for x in hex_to_rgb(hex_color)]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    harmonies = set()

    # Add the original color
    harmonies.add(hex_color)

    # 1. Complementary
    comp_h = (h + 0.5) % 1.0
    harmonies.add(rgb_to_hex([int(x * 255) for x in colorsys.hsv_to_rgb(comp_h, s, v)]))

    # 2. Analogous
    for i in [-1, 1]:
        ana_h = (h + i * (30 / 360.0)) % 1.0
        harmonies.add(rgb_to_hex([int(x * 255) for x in colorsys.hsv_to_rgb(ana_h, s, v)]))

    # 3. Monochromatic
    harmonies.add(rgb_to_hex([int(x * 255) for x in colorsys.hsv_to_rgb(h, max(0, s - 0.3), min(1, v + 0.3))]))
    harmonies.add(rgb_to_hex([int(x * 255) for x in colorsys.hsv_to_rgb(h, s, max(0, v - 0.4))]))

    # 4. Key Neutrals
    harmonies.update(["#000000", "#FFFFFF", "#808080", "#bebebe", "#f5f5dc", "#343434"])
    
    return list(harmonies)

def find_best_matches(ideal_colors_hex, available_colors_hex):
    """
    For each available color, finds its closest match from the ideal colors list
    and returns a ranked list of available colors based on that distance.
    """
    if not available_colors_hex or not ideal_colors_hex:
        return []

    # Convert ideal colors to LAB once to be efficient
    ideal_labs = []
    for hex_code in ideal_colors_hex:
        rgb = np.array(hex_to_rgb(hex_code), dtype=np.uint8)
        ideal_labs.append(rgb2lab(rgb.reshape(1, 1, 3)))

    ranked_available = []
    for avail_hex in available_colors_hex:
        avail_rgb = np.array(hex_to_rgb(avail_hex), dtype=np.uint8)
        avail_lab = rgb2lab(avail_rgb.reshape(1, 1, 3))

        # Find the minimum distance to any of the ideal colors
        min_distance = float('inf')
        for ideal_lab in ideal_labs:
            distance = deltaE_cie76(avail_lab, ideal_lab)
            if distance < min_distance:
                min_distance = distance
        
        ranked_available.append((avail_hex, float(min_distance)))

    # Sort by the minimum distance (lower is better)
    ranked_available.sort(key=lambda x: x[1])
    return ranked_available


# --- Command-Line Interface ---
if __name__ == "__main__":
    print("--- Outfit Color Coordinator ---")
    
    categories = ["top", "bottom", "shoes", "accessory"]
    
    # 1. Get the primary clothing item
    primary_item = ""
    while primary_item not in categories:
        primary_item = input(f"What is your primary clothing item? ({', '.join(categories)}): ").lower()

    primary_color = ""
    while not is_valid_hex(primary_color):
        primary_color = input(f"Enter the HEX color of your {primary_item} (e.g., #ff5733): ")
        if not is_valid_hex(primary_color):
            print("Invalid HEX code format. Please try again.")

    # 2. Get available colors for the other categories
    available_wardrobe = {}
    for category in categories:
        if category == primary_item:
            continue # Skip the item we're building around
            
        color_list_input = input(f"\nEnter available HEX colors for '{category}', separated by commas:\n")
        raw_colors = [color.strip() for color in color_list_input.split(',')]
        valid_colors = [color for color in raw_colors if is_valid_hex(color)]
        available_wardrobe[category] = valid_colors

    # 3. Generate ideal harmony colors based on the primary item
    ideal_colors = get_ideal_harmony_colors(primary_color)
    
    # 4. Find and display the best matches from the user's wardrobe
    print("\n" + "="*50)
    print(f"Top 5 Outfit Recommendations to go with {primary_color} {primary_item}:")
    print("(Lower 'Harmony Score' means a better match)")
    print("="*50)

    for category, color_list in available_wardrobe.items():
        print(f"\n--- Recommended for {category.capitalize()} ---")
        if not color_list:
            print("No colors were provided for this category.")
            continue

        ranked_choices = find_best_matches(ideal_colors, color_list)
        
        if not ranked_choices:
            print("Could not rank the provided colors.")
            continue
        
        for i, (color, score) in enumerate(ranked_choices[:5]):
            print(f"{i+1}. {color} (Harmony Score: {score:.2f})")

