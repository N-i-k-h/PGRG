# PGRG Steel Defect Detection

**Priority-Guided Region Growing for Fast Surface Defect Detection in Steel Strip Images**

## Overview

This project implements a Priority-Guided Region Growing (PGRG) algorithm
for detecting surface defects on steel strips. The algorithm uses texture
variance and gradient magnitude to build a priority map, then performs
region growing guided by a max-heap data structure.

## Dataset

This project uses the **NEU Surface Defect Database (NEU-DET)**.

- **Total images:** 1,800 grayscale images (200 × 200 pixels)
- **Training set:** 1,440 images
- **Validation set:** 360 images
- **Six defect classes:**
  1. Crazing
  2. Inclusion
  3. Patches
  4. Pitted surface
  5. Rolled-in scale
  6. Scratches

Place the dataset in `dataset/NEU-DET/` with the following structure:

```
dataset/NEU-DET/
├── train/
│   ├── images/
│   │   ├── crazing/
│   │   ├── inclusion/
│   │   ├── patches/
│   │   ├── pitted_surface/
│   │   ├── rolled-in_scale/
│   │   └── scratches/
│   └── annotations/
└── validation/
    ├── images/
    │   ├── crazing/
    │   ├── inclusion/
    │   ├── patches/
    │   ├── pitted_surface/
    │   ├── rolled-in_scale/
    │   └── scratches/
    └── annotations/
```

## Project Structure

```
PGRG-Steel-Defect-Detection/
├── dataset/NEU-DET/          # Dataset (not included in repo)
├── src/                      # Source code modules
│   ├── preprocessing.py      # Image preprocessing
│   ├── priority_features.py  # Priority feature extraction
│   ├── priority_map.py       # Priority map generation
│   ├── union_find.py         # Union-Find data structure
│   ├── pgrg.py               # Main PGRG algorithm
│   ├── region_growing.py     # Region growing segmentation
│   ├── otsu.py               # Otsu baseline method
│   ├── watershed.py          # Watershed baseline method
│   └── metrics.py            # Evaluation metrics
├── results/                  # Output results
│   ├── masks/                # Segmentation masks
│   ├── priority_maps/        # Priority map visualizations
│   └── figures/              # Plots and figures
├── experiments/              # Experiment configurations and logs
├── main.py                   # Entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1. Create and activate virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Linux/macOS
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place the NEU-DET dataset in `dataset/NEU-DET/`.

4. Run the project:
   ```
   python main.py
   ```


