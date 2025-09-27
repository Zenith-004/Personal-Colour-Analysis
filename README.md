# Personal-Colour-Analysis
Python-based command-line tools for personal color analysis and outfit coordination. Analyze your color season, detect clothing colors, get outfit recommendations, and rank colors for harmony. Powered by OpenCV, dlib, and color theory.
---

# 🎨 Personal Color & Outfit Command-Line Tools

A suite of **Python-based command-line tools** for personal color analysis and outfit coordination.
You can analyze your personal color season, identify the color of clothing items, and get recommendations for building harmonious outfits.

---

## 📖 Project Overview

This collection includes **four main scripts**:

1. **`color_analysis.py`**
   Analyzes an image of a person to determine their **12-season personal color type** (e.g., Deep Winter, Soft Autumn).

2. **`clothing_analyzer.py`**
   Analyzes an image of a clothing item to identify its **primary colors and properties**.

3. **`outfit_coordinator.py`**
   A command-line tool to help you **build a coordinated outfit** around a specific clothing item and color.

4. **`color_ranker.py`**
   A utility to **rank a list of colors** based on their perceptual closeness to a target color.

---

## ⚙️ Prerequisites & Installation

### Step 1: Install C++ Compiler and CMake (required for `dlib`)

The **dlib** library requires a C++ compiler and the CMake build tool.

* **Windows**:

  1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
     In the installer, select:

     * ✅ Desktop development with C++
     * ✅ Windows 10/11 SDK
     * ✅ MSVC v14.x toolset
  2. Install [CMake](https://cmake.org/download/).
     During installation, check:
     *“Add CMake to the system PATH for all users”*

---

### Step 2: Install Python Libraries

Once build tools are ready, open your terminal and run:    
[NOTE: IT MIGHT TAKE SOME TIME]

```bash
pip install opencv-python dlib scikit-image numpy
```

---

### Step 3: Download dlib Shape Predictor Model
[Note: Download will automatically start in 5 Sec after cliking link]

The `color_analysis.py` script needs a pre-trained facial landmark detector.

1. Download **[shape_predictor_68_face_landmarks.dat]([http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2))**
2. Extract it (use [7-Zip](https://www.7-zip.org/) or WinRAR).
3. Place the `.dat` file in the same directory as your Python scripts.

---

## 🚀 How to Use

### 1. `color_analysis.py` — Personal Color Analysis

Analyzes a photo to determine your **color season**.

```bash
python color_analysis.py
```
Enter image relative path like this: path/to/person/image.png

**Example Output:**

```
--- Personal Color Analysis ---
Analyzing image: path/to/your/image.jpg

--- Analysis Results ---
Detected Skin Color (RGB): (238, 193, 169) | HEX: #eec1a9
Detected Hair Color (RGB): (23, 19, 13)    | HEX: #17130d
Detected Eye Color (RGB): (85, 63, 48)     | HEX: #553f30

--- Classification ---
Skin Undertone: Warm
Contrast Level: High
Saturation Level: Muted

--- Final Determination ---
Determined Season: Deep Autumn

--- Recommended Palette ---
Good Colors: ['#8B4513', '#A0522D', '#B22222', '#8B0000', '#CD853F', ...]
Bad Colors:  ['#00CED1', '#00FFFF', '#7FFFD4', '#1E90FF', '#E6E6FA']
```

---

### 2. `clothing_analyzer.py` — Clothing Color Analyzer

Identifies the **main color(s)** of a piece of clothing in an image.

```bash
python clothing_analyzer.py
```
Enter image relative path like this: path/to/clothing/image.png

**Example Output:**

```
--- Clothing Color Analysis ---
Analyzing image: path/to/clothing/image.png

--- Dominant Colors Found ---
1. #2a3b8f (Primary)
2. #273682
3. #f7f7f7

--- Primary Color Details ---
HEX Code: #2a3b8f
RGB Value: (42, 59, 143)
HSV Value: (230, 70, 56)
Color Temperature: Cool

--- Suggested For Seasons ---
This color is flattering for:
- Deep Winter
- Cool Winter
- Clear Winter
- Cool Summer
```

---

### 3. `outfit_coordinator.py` — Outfit Recommendations

Suggests **color pairings** based on your clothing item.

```bash
python outfit_coordinator.py
```

**Example Interaction:**

```
--- Outfit Color Coordinator ---
What is your primary clothing item? (top, bottom, shoes, accessory): top
Enter the HEX color of your top (e.g., #ff5733): #33a8ff

--- Outfit Recommendations for a #33a8ff Top ---

Complementary Color (for a bold look): #ff8033
Analogous Colors (for a harmonious look): #33ffda, #337bff
Triadic Colors (for a vibrant look): #ff33a8, #a8ff33

Suggested Pairings:
- For Bottoms: Consider neutrals like #343434 (charcoal) or #f5f5dc (beige), or the complementary #ff8033.
- For Shoes: #000000 (black), #ffffff (white), or a monochromatic shade like #1c5a8d.
- For Accessories: Gold or silver metals. Bright colors like #ff8033 can work as an accent.
```

---

### 4. `color_ranker.py` — Rank Closest Colors

Ranks a list of colors by **closeness to a target color**.

```bash
python color_ranker.py
```

**Example Interaction:**

```
--- Color Ranker ---
Enter the target HEX color (e.g., #ff5733): #ff6b33
Enter a list of HEX colors to rank, separated by commas:
#ff5733, #e64e1c, #d3ab9a, #33a8ff

========================================
Colors ranked by closeness to #ff6b33:
(Lower score means a closer match)
========================================

1. #ff5733 (Score: 6.91)
2. #e64e1c (Score: 11.23)
3. #d3ab9a (Score: 34.50)
4. #33a8ff (Score: 78.14)
```

---

## 📌 Notes

* Works best with **clear, well-lit images** otherwise will show error.
* Sample pictures are provide use similar pictures to gain optimal result.
* Clothing background should be **plain or minimal** for better analysis.
* Recommended for **fashion enthusiasts, stylists, and anyone curious about personal color theory**.

---
