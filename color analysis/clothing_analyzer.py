# Clothing Color Analyzer
# This script analyzes an image of a clothing item to determine its dominant color
# and suggests which of the 12 personal color seasons it is best suited for.

# Installation:
# This script requires the same libraries as the personal color analysis script.
# pip install opencv-python numpy scikit-learn

import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

# --- Helper Functions ---

def get_dominant_colors(image, k=5, num_colors=3):
    """
    Finds the dominant colors in an image using K-Means clustering.
    Returns a list of the top `num_colors` as (BGR, HEX, percentage).
    """
    # Reshape the image to be a list of pixels and convert to float
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)

    # Perform K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Calculate the percentage of each cluster
    _, counts = np.unique(labels, return_counts=True)
    total_pixels = pixels.shape[0]
    percentages = counts / total_pixels

    # Pair colors with their percentages and sort
    color_data = sorted(zip(centers, percentages), key=lambda x: x[1], reverse=True)

    # Format the output
    dominant_colors = []
    for i in range(min(num_colors, len(color_data))):
        bgr_color = np.uint8(color_data[i][0])
        hex_color = rgb_to_hex(bgr_color[::-1])
        percentage = color_data[i][1]
        dominant_colors.append((bgr_color, hex_color, percentage))
        
    return dominant_colors


def rgb_to_hex(rgb_color):
    """Converts an RGB tuple to a HEX color string."""
    r, g, b = int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])
    return f'#{r:02x}{g:02x}{b:02x}'

def analyze_clothing_color(color_bgr):
    """
    Analyzes a BGR color and determines which seasons it suits best based on
    its properties in the HSV color space. Returns a dictionary of properties.
    """
    # Convert BGR to HSV
    hsv_color = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    hue, saturation, value = hsv_color[0], hsv_color[1], hsv_color[2]

    # Normalize hue to 0-360 degrees for easier understanding
    hue_deg = hue * 2

    suitable_seasons = set()
    properties = []

    # Determine Temperature (Warm/Cool)
    is_warm = (hue_deg <= 180)
    is_cool = not is_warm
    temperature = "Warm" if is_warm else "Cool"

    # --- Classification Logic ---

    # Soft/Muted Colors (Low Saturation)
    if saturation < 90:
        properties.append("Soft")
        if is_warm: suitable_seasons.add("Soft Autumn")
        if is_cool: suitable_seasons.add("Soft Summer")

    # Light Colors (High Value, Low-Mid Saturation)
    if value > 200 and saturation < 150:
        properties.append("Light")
        if is_warm: suitable_seasons.add("Light Spring")
        if is_cool: suitable_seasons.add("Light Summer")
    
    # Deep/Dark Colors (Low Value)
    if value < 85:
        properties.append("Deep")
        if is_warm: suitable_seasons.add("Deep Autumn")
        if is_cool: suitable_seasons.add("Deep Winter")

    # Bright/Clear Colors (High Saturation)
    if saturation > 150:
        properties.append("Bright")
        if is_warm: suitable_seasons.add("Bright Spring")
        if is_cool: suitable_seasons.add("Clear Winter")
        
    # General Warm Colors
    if is_warm and saturation > 90 and value > 85:
        suitable_seasons.add("Warm Autumn")
        suitable_seasons.add("Warm Spring")

    # General Cool Colors
    if is_cool and saturation > 90 and value > 85:
        suitable_seasons.add("Cool Winter")
        suitable_seasons.add("Cool Summer")

    final_seasons = sorted(list(suitable_seasons))
    if not final_seasons:
        final_seasons = ["This color is fairly neutral and could work for many seasons."]

    return {
        "hsv": (int(hue), int(saturation), int(value)),
        "temperature": temperature,
        "properties": properties if properties else ["Neutral"],
        "seasons": final_seasons
    }


# --- Command-Line Interface ---
if __name__ == "__main__":
    print("--- Clothing Color Analyzer ---")
    image_path = input("Enter the path to the clothing image file: ").strip()

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'.")
    else:
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not read the image file.")
        else:
            # Get dominant colors
            dominant_colors = get_dominant_colors(image, k=8, num_colors=3)
            
            if not dominant_colors:
                print("Could not determine a dominant color.")
                exit()

            # If multiple colors are found, analyze the second most dominant one to avoid the background.
            if len(dominant_colors) > 1:
                print("\nNote: Analyzing the second most dominant color to likely ignore the background.")
                primary_color_bgr, primary_hex, _ = dominant_colors[1]
            else:
                print("\nNote: Only one dominant color found. Analyzing it.")
                primary_color_bgr, primary_hex, _ = dominant_colors[0]

            primary_color_rgb = primary_color_bgr[::-1]

            # Analyze the primary color
            analysis = analyze_clothing_color(primary_color_bgr)

            # Print results
            print("\n--- Clothing Analysis Results ---")
            
            print("\nPrimary Color Details (analyzed color):")
            print(f"  - RGB: {tuple(primary_color_rgb)}")
            print(f"  - HEX: {primary_hex}")
            print(f"  - HSV (Hue, Saturation, Value): {analysis['hsv']}")
            print(f"  - Temperature: {analysis['temperature']}")
            print(f"  - Properties: {', '.join(analysis['properties'])}")
            
            print("\nDetected Color Range (Top 3):")
            for _, hex_code, percentage in dominant_colors:
                print(f"  - {hex_code} (~{percentage:.1%})")

            print("\n" + "-" * 20)
            print("This color would be most suitable for the following seasons:")
            for season in analysis['seasons']:
                print(f"- {season}")