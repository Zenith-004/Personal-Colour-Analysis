# Personal Color Analysis Program
# This script analyzes an uploaded image to determine a person's skin undertone,
# overall color season, and suggests flattering colors.

# Installation:
# 1. Make sure you have Python installed.
# 2. You will need to install several Python libraries. Open your terminal or command prompt and run:
#    pip install opencv-python dlib numpy scikit-learn
# 3. Download the facial landmark predictor model from:
#    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#    Unzip the file and place 'shape_predictor_68_face_landmarks.dat' in the same folder as this script.

import cv2
import dlib
import numpy as np
from sklearn.cluster import KMeans
import os

# --- Configuration ---
# Path to the dlib facial landmark predictor
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# --- Helper Functions ---

def get_dominant_color(image, k=4):
    """
    Finds the dominant color in an image patch using K-Means clustering.
    Returns the dominant color in BGR format.
    """
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)

    # Perform K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Find the most frequent cluster
    _, counts = np.unique(labels, return_counts=True)
    dominant = centers[np.argmax(counts)]

    return np.uint8(dominant)

def get_skin_undertone(skin_color_bgr):
    """
    Determines if the skin undertone is warm, cool, or neutral.
    Analyzes the color in the Lab color space.
    """
    # Convert BGR to Lab color space
    lab_image = cv2.cvtColor(np.uint8([[skin_color_bgr]]), cv2.COLOR_BGR2Lab)[0][0]
    # The 'b' channel in Lab represents the blue-yellow axis.
    # Higher 'b' value means more yellow (warm), lower means more blue (cool).
    b_value = float(lab_image[2])

    if b_value > 128:  # Threshold can be adjusted
        return "Warm"
    elif b_value < 120:
        return "Cool"
    else:
        return "Neutral"

def rgb_to_hex(rgb_color):
    """Converts an RGB tuple to a HEX color string."""
    # Ensure color values are integers
    r, g, b = int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])
    return f'#{r:02x}{g:02x}{b:02x}'

# --- Main Analysis Function ---

def analyze_image(image_path):
    """
    Performs the full color analysis on a given image.
    """
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        return {"error": f"Dlib shape predictor not found at '{SHAPE_PREDICTOR_PATH}'. Please download it."}
    if not os.path.exists(image_path):
        return {"error": f"Image file not found at '{image_path}'."}

    # Load the detector and predictor
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    except Exception as e:
        return {"error": f"Failed to load dlib models. Error: {e}"}

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not read the image file. It might be corrupted or in an unsupported format."}

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return {"error": "No faces detected in the image. Please use a clear, front-facing photo."}

    face = faces[0] # Use the first detected face
    landmarks = predictor(gray, face)

    # --- Color Extraction ---
    # Skin color from cheeks
    left_cheek_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in [2, 3, 4, 31, 41]])
    right_cheek_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in [14, 13, 12, 35, 47]])
    
    left_cheek_roi = cv2.boundingRect(left_cheek_points)
    right_cheek_roi = cv2.boundingRect(right_cheek_points)
    
    skin_patch = image[left_cheek_roi[1]:left_cheek_roi[1]+left_cheek_roi[3], left_cheek_roi[0]:left_cheek_roi[0]+left_cheek_roi[2]]
    skin_color_bgr = get_dominant_color(skin_patch)

    # Hair color from top of the head
    face_rect = (face.left(), face.top(), face.width(), face.height())
    hair_y = max(0, face.top() - int(face.height() * 0.3))
    hair_patch = image[hair_y:face.top(), face.left():face.right()]
    if hair_patch.size == 0:
        hair_color_bgr = np.array([0, 0, 0]) # Default to black if region is invalid
    else:
        hair_color_bgr = get_dominant_color(hair_patch, k=3)

    # Eye color
    left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    eye_roi = cv2.boundingRect(left_eye_points)
    eye_patch = image[eye_roi[1]:eye_roi[1]+eye_roi[3], eye_roi[0]:eye_roi[0]+eye_roi[2]]
    eye_color_bgr = get_dominant_color(eye_patch, k=3)

    # --- Analysis ---
    skin_undertone = get_skin_undertone(skin_color_bgr)
    
    # Convert BGR to RGB for analysis and display
    skin_color_rgb = skin_color_bgr[::-1]
    hair_color_rgb = hair_color_bgr[::-1]
    eye_color_rgb = eye_color_bgr[::-1]
    
    # Convert to HSV for lightness/darkness and saturation checks
    skin_color_hsv = cv2.cvtColor(np.uint8([[skin_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    hair_color_hsv = cv2.cvtColor(np.uint8([[hair_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    
    skin_saturation = skin_color_hsv[1]
    skin_lightness = skin_color_hsv[2]
    hair_lightness = hair_color_hsv[2]
    
    contrast = abs(skin_lightness - hair_lightness)
    
    # --- 12-Season Classification Logic ---
    season = "Undetermined"
    if skin_undertone == "Cool":
        if contrast > 100 and hair_lightness < 70:  # High contrast & dark hair -> Winter
            if contrast > 120:
                season = "Clear Winter"
            elif hair_lightness < 40:
                season = "Deep Winter"
            else:
                season = "Cool Winter"
        else:  # Low-mid contrast -> Summer
            if skin_lightness > 190 and hair_lightness > 120:
                season = "Light Summer"
            elif skin_saturation < 85:
                season = "Soft Summer"
            else:
                season = "Cool Summer"
    elif skin_undertone == "Warm":
        if hair_lightness < 100: # Darker hair -> Autumn
            if hair_lightness < 50 and contrast > 80:
                season = "Deep Autumn"
            elif skin_saturation < 90:
                season = "Soft Autumn"
            else:
                season = "Warm Autumn"
        else: # Lighter hair -> Spring
            if skin_lightness > 190:
                season = "Light Spring"
            elif contrast > 90:
                 season = "Bright Spring"
            else:
                 season = "Warm Spring"
    else: # Neutral undertone logic
        if skin_saturation < 80: # Likely soft season
            season = "Soft Autumn" if hair_lightness < 110 else "Soft Summer"
        elif contrast > 110: # Likely clear/bright season
            season = "Clear Winter" if hair_lightness < 60 else "Bright Spring"
        else: # Fallback for neutral
            season = "Deep Autumn" if hair_lightness < 60 else "Warm Spring"

    # --- Expanded 12-Season Color Palettes for Personal Color Analysis ---
    palettes = {
        # --- WINTER VARIANTS ---
        "Deep Winter": {
            "good": [
                "#000000", "#FFFFFF", "#2F4F4F", "#800000", "#8B0000",
                "#000080", "#483D8B", "#008080", "#4B0082", "#191970",
                "#4169E1", "#228B22", "#9932CC", "#B22222", "#4682B4"
            ],
            "bad": [
                "#F4A460", "#DEB887", "#DAA520", "#FFDAB9", "#FFE4B5"
            ]
        },
        "Cool Winter": {
            "good": [
                "#000000", "#FFFFFF", "#4682B4", "#00CED1", "#008B8B",
                "#5F9EA0", "#6495ED", "#7B68EE", "#C71585", "#FF1493",
                "#9932CC", "#B0C4DE", "#87CEEB", "#00BFFF", "#9400D3"
            ],
            "bad": [
                "#FF8C00", "#D2691E", "#F4A460", "#CD853F", "#B8860B"
            ]
        },
        "Clear Winter": {
            "good": [
                "#000000", "#FFFFFF", "#00BFFF", "#FF1493", "#FF0000",
                "#00FA9A", "#7FFFD4", "#9400D3", "#FF4500", "#DC143C",
                "#1E90FF", "#FF69B4", "#00FFFF", "#8A2BE2", "#FF6347"
            ],
            "bad": [
                "#808080", "#696969", "#C0C0C0", "#D3D3D3"
            ]
        },

        # --- SUMMER VARIANTS ---
        "Light Summer": {
            "good": [
                "#E6E6FA", "#AFEEEE", "#87CEFA", "#D8BFD8", "#ADD8E6",
                "#FFE4E1", "#B0E0E6", "#F0E68C", "#98FB98", "#DB7093",
                "#C0C0C0", "#BA55D3", "#F08080", "#F5F5DC", "#C8A2C8"
            ],
            "bad": [
                "#000000", "#8B0000", "#4B0082", "#2F4F4F", "#FF8C00"
            ]
        },
        "Cool Summer": {
            "good": [
                "#4682B4", "#6495ED", "#B0C4DE", "#9370DB", "#E6E6FA",
                "#87CEEB", "#708090", "#20B2AA", "#778899", "#6A5ACD",
                "#C71585", "#DDA0DD", "#A9A9A9", "#48D1CC", "#87CEFA"
            ],
            "bad": [
                "#FF7F50", "#FF4500", "#FFD700", "#FFA500", "#B22222"
            ]
        },
        "Soft Summer": {
            "good": [
                "#B0C4DE", "#D8BFD8", "#C0C0C0", "#E0FFFF", "#98FB98",
                "#AFEEEE", "#DB7093", "#778899", "#708090", "#D3D3D3",
                "#BC8F8F", "#C8A2C8", "#87CEEB", "#C71585", "#90EE90"
            ],
            "bad": [
                "#FF0000", "#FF00FF", "#FFFF00", "#000000", "#FFA500"
            ]
        },

        # --- AUTUMN VARIANTS ---
        "Deep Autumn": {
            "good": [
                "#8B4513", "#A0522D", "#B22222", "#8B0000", "#CD853F",
                "#D2691E", "#C19A6B", "#B8860B", "#FF8C00", "#A52A2A",
                "#6B8E23", "#556B2F", "#BC8F8F", "#8B008B", "#BDB76B"
            ],
            "bad": [
                "#00CED1", "#00FFFF", "#7FFFD4", "#1E90FF", "#E6E6FA"
            ]
        },
        "Warm Autumn": {
            "good": [
                "#FF8C00", "#D2691E", "#F4A460", "#CD853F", "#B8860B",
                "#DAA520", "#DEB887", "#C19A6B", "#8B4513", "#FFD700",
                "#808000", "#9ACD32", "#556B2F", "#8B0000", "#6B4226"
            ],
            "bad": [
                "#000000", "#FFFFFF", "#FF00FF", "#00BFFF", "#7B68EE"
            ]
        },
        "Soft Autumn": {
            "good": [
                "#C19A6B", "#CD853F", "#D2B48C", "#BC8F8F", "#DAA520",
                "#9ACD32", "#BDB76B", "#DEB887", "#F4A460", "#6B8E23",
                "#8B4513", "#C0C0C0", "#EEE8AA", "#C71585", "#8B0000"
            ],
            "bad": [
                "#0000FF", "#FF00FF", "#00FFFF", "#FF69B4", "#E0FFFF"
            ]
        },

        # --- SPRING VARIANTS ---
        "Bright Spring": {
            "good": [
                "#FFD700", "#FF7F50", "#FF4500", "#FF69B4", "#87CEEB",
                "#00FA9A", "#7CFC00", "#FF1493", "#00FFFF", "#FFA500",
                "#00CED1", "#ADFF2F", "#FFB6C1", "#F0E68C", "#98FB98"
            ],
            "bad": [
                "#808080", "#696969", "#2F4F4F", "#4B0082", "#191970"
            ]
        },
        "Warm Spring": {
            "good": [
                "#FFD700", "#FFA500", "#FF8C00", "#F0E68C", "#EEE8AA",
                "#98FB98", "#00FA9A", "#ADFF2F", "#87CEEB", "#40E0D0",
                "#FF69B4", "#FFB6C1", "#87CEFA", "#FF6347", "#FA8072"
            ],
            "bad": [
                "#000000", "#2F4F4F", "#4B0082", "#8B0000", "#808080"
            ]
        },
        "Light Spring": {
            "good": [
                "#FFDAB9", "#FFE4B5", "#FAFAD2", "#FFFACD", "#F5DEB3",
                "#98FB98", "#AFEEEE", "#E0FFFF", "#FFE4E1", "#FFD700",
                "#FFB6C1", "#FF69B4", "#87CEEB", "#ADFF2F", "#FF7F50"
            ],
            "bad": [
                "#000000", "#800000", "#191970", "#4B0082", "#2F4F4F"
            ]
        },

        # --- FALLBACK ---
        "Undetermined": {
            "good": ["#808080", "#C0C0C0", "#D3D3D3", "#A9A9A9"],
            "bad": ["#808080"]
        }
    }


    return {
        "skin_color_rgb": tuple(skin_color_rgb.astype(int)),
        "skin_color_hex": rgb_to_hex(skin_color_rgb),
        "hair_color_rgb": tuple(hair_color_rgb.astype(int)),
        "hair_color_hex": rgb_to_hex(hair_color_rgb),
        "eye_color_rgb": tuple(eye_color_rgb.astype(int)),
        "eye_color_hex": rgb_to_hex(eye_color_rgb),
        "skin_undertone": skin_undertone,
        "season": season,
        "recommended_colors": palettes[season]["good"],
        "colors_to_avoid": palettes[season]["bad"]
    }

# --- Command-Line Interface ---
if __name__ == "__main__":
    print("--- Personal Color Analysis ---")
    
    # Check for the shape predictor file first
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        print(f"\nERROR: Dlib shape predictor model not found.")
        print(f"Please download '{SHAPE_PREDICTOR_PATH}' and place it in the same directory as this script.")
        print("You can download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    else:
        image_path = input("Enter the path to the image file of a person: ").strip()
        
        results = analyze_image(image_path)
        
        print("\n--- Analysis Results ---")
        if "error" in results:
            print(f"An error occurred: {results['error']}")
        else:
            print(f"Detected Skin Color | RGB: {results['skin_color_rgb']} | HEX: {results['skin_color_hex']}")
            print(f"Detected Hair Color | RGB: {results['hair_color_rgb']} | HEX: {results['hair_color_hex']}")
            print(f"Detected Eye Color  | RGB: {results['eye_color_rgb']} | HEX: {results['eye_color_hex']}")
            print("-" * 20)
            print(f"Skin Undertone: {results['skin_undertone']}")
            print(f"Determined Season: {results['season']}")
            print("-" * 20)
            print("Recommended Colors:")
            print("  " + ", ".join(results['recommended_colors']))
            print("\nColors to Avoid:")
            print("  " + ", ".join(results['colors_to_avoid']))