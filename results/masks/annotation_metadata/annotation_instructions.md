# User Guide: Manual Steel Defect Annotation Tool (`tools/annotate_masks.py`)

## 1. Overview & Objectives

In standard defect detection benchmarks (such as NEU-DET), annotations are provided exclusively as rectangular bounding boxes (`<bndbox>` in Pascal VOC XML). While bounding boxes suffice for coarse object localization, they encompass substantial areas of benign background metal. 

To enable **genuine pixel-level segmentation evaluation** without hallucinating ground-truth masks, this manual annotation tool allows researchers to draw exact polygon contours directly around defect pixels.

> [!IMPORTANT]
> **Strict Separation from Approximate ROI Masks:**
> Rectangular bounding boxes must **never** be rasterized as exact pixel-level ground truth. In our empirical measurements, rectangular bounding boxes overestimate true defect pixel area by up to **335%** (e.g., `scratches_4.jpg` has 1,714 genuine defect pixels vs. 5,754 rectangular ROI pixels). Any generated bounding-box masks in `results/masks/approximate_roi/` are flagged as approximate and are strictly excluded from genuine pixel-level evaluation.

---

## 2. Guidelines for Selecting Representative Images

When selecting defect samples for the 30-image pilot dataset (5 images per class across all 6 classes), follow these class-specific criteria:

| Defect Class | Visual Characteristics | Selection & Delineation Guidance |
|:---|:---|:---|
| **Scratches** | Sharp, high-aspect-ratio longitudinal trenches | Select samples with distinct single and intersecting scratch grooves. Trace along the bright/dark scratch track; do not include surrounding flat steel. |
| **Inclusion** | Compact, dark non-metallic flecks | Choose samples with isolated and clustered inclusions. Trace the dark core anomaly, avoiding benign rolling grain lines. |
| **Crazing** | Complex network of fine micro-crack fissures | Select samples with varied fissure densities. Trace primary crack branches; avoid labeling flat inter-crack metal. |
| **Pitted Surface** | Dense micro-cavities and rough surface pitting | Select surfaces with localized and dispersed pitting. Delineate pit clusters while excluding benign flat background plateaus. |
| **Patches** | Broad plate-like surface irregularities and oxide films | Select images showing clear contrast boundaries between the patch layer and the base substrate. |
| **Rolled-in Scale** | Embedded mill scale streaks and iron oxide indentations | Select images with elongated scale impressions. Delineate dark scale streaks while preserving underlying steel boundaries. |

---

## 3. How to Launch and Use the Annotation Tool

### Launch Command
To launch the interactive OpenCV annotation GUI:
```bash
python tools/annotate_masks.py
```

### Display Scaling
By default, the tool renders $200 \times 200$ images scaled at **3x** ($600 \times 600$ pixels) so that fine micro-features can be accurately clicked. To use a 4x scale ($800 \times 800$ pixels):
```bash
python tools/annotate_masks.py --scale 4
```

### Interactive Controls

| Key / Action | Function | Description |
|:---|:---|:---|
| **Left Click** | Add Vertex | Adds a vertex to the current in-progress polygon. Yellow dots and lines show the current contour. |
| **Right Click** or **`u`** | Undo | Removes the most recent vertex. If current polygon is empty, un-closes the last completed polygon. |
| **`c`** | Complete Polygon | Closes the polygon (requires $\ge 3$ vertices) and fills it with translucent green overlay. Multiple disjoint polygons can be drawn! |
| **`r`** | Reset Image | Discards all polygons on the current image to start over. |
| **`b`** | Toggle XML Box | Displays the PASCAL VOC bounding box in dashed orange with a prominent watermark as a rough location reference only. |
| **`s`** | Save & Next | Rasterizes all completed polygons into a single-channel 8-bit PNG (`0`=background, `255`=defect), saves to `results/masks/ground_truth/<filename>.png`, updates the metadata manifest, and loads the next image. |
| **`n`** or **`d`** | Next Image | Advances to next image without saving. |
| **`p`** or **`a`** | Previous Image | Returns to previous image in the pilot dataset. |
| **`q`** or **ESC** | Quit Session | Safely exits the annotator and saves progress tracking JSON. |

---

## 4. Metadata and Dataset Tracking

All annotations and progress metrics are tracked in JSON format under `results/masks/annotation_metadata/`:
- `pilot_dataset_manifest.json`: Full manifest of all 30 pilot images, tracking image path, XML path, mask path, and annotation status (`annotated` vs `pending_human_annotation`).
- `annotation_manifest.json`: Detailed polygon counts, defect pixel counts, and timestamps for saved masks.
- `annotation_progress.json`: Real-time summary of completed vs pending images.

---

## 5. Re-running the Pixel Evaluation Pipeline

Whenever additional images are annotated and saved, re-run the pixel-level evaluation script to update all CSV summaries, comparison plots, and reports:
```bash
python experiments/run_pixel_evaluation.py
```
This script will evaluate all completed masks in `results/masks/ground_truth/` and automatically regenerate:
- `results/pixel_evaluation/pixel_metrics_per_image.csv`
- `results/pixel_evaluation/pixel_metrics_summary.csv`
- `results/pixel_evaluation/pixel_evaluation_comparison.png`
- `results/pixel_evaluation/pixel_evaluation_report.md`
