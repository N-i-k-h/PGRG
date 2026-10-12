# PGRG Steel Defect Detection

**Priority-Guided Region Growing for Fast Surface Defect Detection in Steel Strip Images**

---

## 1. Overview

This project implements the **Priority-Guided Region Growing (PGRG)** algorithm for detecting surface defects on hot-rolled and cold-rolled steel strips.

The algorithm extracts three multi-scale visual cues:
1. **Intensity Deviation** ($|I(x, y) - \mu_{\text{local}}|$)
2. **Sobel Gradient Magnitude** ($\sqrt{G_x^2 + G_y^2}$)
3. **Local Texture Variance** ($\mu(I^2) - \mu(I)^2$)

These features are normalized and combined into a continuous priority map. A max-heap data structure then guides region expansion along prominent defect structures, followed by Union-Find disjoint-set connectivity analysis and area filtering.

> [!NOTE]
> **Improved PGRG is the Default Algorithm:**  
> The project now defaults to the verified **Improved PGRG** configuration, which achieves a **+91.5% F1 gain** and a **72.0% reduction in false alarms** compared to the original baseline while preserving high localization accuracy ($\text{IoU} \approx 0.70$).

---

## 2. Algorithm Configurations

| Parameter | Improved PGRG (Default) | Original PGRG (Baseline) | Optimization Purpose |
|---|---|---|---|
| `min_area` | **100 px** | 20 px | Filters high-frequency background noise and small spurious candidates. |
| `seed_threshold` | **0.45** | 0.45 | Optimal balance of sensitivity without seed noise. |
| `similarity_threshold` | **25** | 20 | Enables regions to grow to complete defect boundaries. |
| `merge_gap` | **11 px** | 0 px (None) | $11\times 11$ elliptical closing bridges fragmented linear defects (e.g. scratches). |
| `weights` | **Equal ($\frac{1}{3}, \frac{1}{3}, \frac{1}{3}$)** | Equal ($\frac{1}{3}, \frac{1}{3}, \frac{1}{3}$) | Balanced multi-cue priority map integration. |

Both configurations and all baseline algorithms are accessible programmatically via [`src/pgrg.py`](src/pgrg.py) or through the CLI in [`main.py`](main.py).

---

## 3. Dataset (NEU-DET)

The project benchmarks on the **NEU Surface Defect Database (NEU-DET)** containing **1,800 images** ($200 \times 200$ pixels) across 6 defect classes:
1. **crazing**
2. **inclusion**
3. **patches**
4. **pitted_surface**
5. **rolled-in_scale**
6. **scratches**

Dataset split:
* **Training set:** 1,440 images (`train/images/`, `train/annotations/`)
* **Held-out validation set:** 360 images (`validation/images/`, `validation/annotations/`)

Directory structure:
```
dataset/NEU-DET/
├── train/
│   ├── images/
│   │   ├── crazing/ ... scratches/
│   └── annotations/
└── validation/
    ├── images/
    │   ├── crazing/ ... scratches/
    └── annotations/
```

---

## 4. Benchmark Performance (360 Held-Out Validation Images)

Evaluation uses bounding-box localization with an $\text{IoU} \ge 0.5$ threshold and Hungarian one-to-one matching:

| Method | TP | FP | FN | Precision | Recall | F1-Score | Mean IoU | Runtime / Image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Improved PGRG (Default)** | **110** | **502** | **744** | **0.1797** | **0.1288** | **0.1501** | **0.6980** | **180.87 ms** |
| Original PGRG | 108 | 1,792 | 746 | 0.0568 | 0.1265 | 0.0784 | 0.6987 | 135.54 ms |
| Standard Region Growing | 79 | 8,419 | 775 | 0.0093 | 0.0925 | 0.0169 | 0.7113 | 95.22 ms |
| Otsu + Morphology | 110 | 7,855 | 744 | 0.0138 | 0.1288 | 0.0249 | 0.6982 | 0.19 ms |
| Watershed | 50 | 2,470 | 804 | 0.0198 | 0.0585 | 0.0296 | 0.7021 | 1.55 ms |

### Class-Wise F1 Scores:
* **pitted_surface:** $0.3333$ (Improved) vs $0.1029$ (Original) — **$+223.9\%$ gain**
* **scratches:** $0.3217$ (Improved) vs $0.1303$ (Original) — **$+146.9\%$ gain**
* **patches:** $0.1655$ (Improved) vs $0.1680$ (Original)
* **inclusion:** $0.0611$ (Improved) vs $0.0537$ (Original)
* **crazing:** $0.0154$ (Improved) vs $0.0179$ (Original)
* **rolled-in_scale:** $0.0090$ (Improved) vs $0.0216$ (Original)

---

## 5. Usage & Commands

### Setup
```bash
python -m venv venv
.\venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Running the Default Pipeline (Improved PGRG)
```bash
# Run Box-Level Evaluation with Improved PGRG by default:
python main.py "dataset/NEU-DET/validation/images/scratches/scratches_241.jpg"

# Run Phase 7 (Final Defect Mask) with Improved PGRG:
python main.py --phase 7 "dataset/NEU-DET/validation/images/scratches/scratches_241.jpg"
```

### Running with Original PGRG Baseline
```bash
# Select Original PGRG baseline using --original-pgrg flag:
python main.py --original-pgrg "dataset/NEU-DET/validation/images/scratches/scratches_241.jpg"

# Run Phase 7 with Original PGRG baseline:
python main.py --phase 7 --original-pgrg "dataset/NEU-DET/validation/images/scratches/scratches_241.jpg"
```

### Running Baseline Methods
```bash
python main.py --baseline1 "dataset/NEU-DET/validation/images/scratches/scratches_241.jpg"  # Standard RG
python main.py --baseline2 "dataset/NEU-DET/validation/images/scratches/scratches_241.jpg"  # Otsu + Morphology
python main.py --baseline3 "dataset/NEU-DET/validation/images/scratches/scratches_241.jpg"  # Watershed
```

### Running Tests
```bash
python -m unittest tests/test_pgrg_integration.py -v
```

---

## 6. Project Limitations

1. **Bounding-Box Ground Truth:** NEU-DET provides bounding-box XML annotations, **not pixel-level segmentation masks**. Evaluation metrics evaluate object-level localization ($\text{IoU} \ge 0.5$).
2. **Challenging Texture Classes:** Fine network cracks (`crazing`) and low-contrast surface scale (`rolled-in_scale`) remain challenging for classical gradient/intensity-based algorithms.
3. **Global Morphological Closing:** The $11\text{ px}$ closing kernel operates uniformly; on crowded adjacent defects, nearby independent instances may be merged into a single envelope.

