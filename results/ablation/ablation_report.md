# Ablation Study Report — PGRG Feature Contribution Analysis

---

## 1. Objective

Measure how each individual priority-map feature contributes to the
overall PGRG defect detection performance. This ablation study isolates
three single-feature variants and compares them against the full
three-feature combination used by the default PGRG configuration.

---

## 2. Configurations

| # | Configuration | Deviation Weight | Gradient Weight | Texture Weight |
|---|---|---|---|---|
| 1 | Deviation Only | **1.0** | 0.0 | 0.0 |
| 2 | Gradient Only | 0.0 | **1.0** | 0.0 |
| 3 | Texture Only | 0.0 | 0.0 | **1.0** |
| 4 | Full PGRG | 1/3 | 1/3 | 1/3 |

The Full PGRG configuration uses `weights=None`, which defaults to equal
weights (1/3 each) in `compute_priority_map()`. This is identical to the
configuration used in all existing benchmark evaluations.

---

## 3. Methodology

### Sample Selection
- **Total images**: 60
- **Images per class**: 10
- **Class distribution**: crazing: 10, inclusion: 10, patches: 10, pitted_surface: 10, rolled-in_scale: 10, scratches: 10
- **Selection**: First 10 images per class, sorted by numeric index
  (deterministic, reproducible)

### Shared Parameters
- Seed threshold: 0.45
- Similarity threshold: 20
- Minimum area: 20
- IoU threshold: 0.5
- Connectivity: 8-neighbour

### Evaluation Protocol
- **Type**: Box-level localization evaluation
- Predicted defect masks are converted to bounding boxes via connected
  component analysis (`mask_to_boxes`).
- Greedy one-to-one matching between predicted and ground-truth boxes
  with IoU ≥ 0.5.
- Metrics: Precision, Recall, F1, Mean Matched IoU.
- Ground truth: PASCAL VOC XML bounding-box annotations from NEU-DET.

### Runtime Measurement
- **Included in runtime**: Priority feature extraction, priority map
  computation, region growing, union-find merging, region filtering.
- **Excluded from runtime**: Image loading, grayscale conversion,
  Gaussian blur, normalization (preprocessing).
- Measured using `time.perf_counter()` for high precision.

> **Note**: These are box-level localization metrics, not pixel-level
> segmentation metrics. NEU-DET provides only bounding-box annotations.

---

## 4. Results

### Overall Performance

| Configuration | Images | TP | FP | FN | Precision | Recall | F1 | Mean IoU | Runtime (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Deviation Only | 60 | 17 | 416 | 107 | 0.0393 | 0.1371 | 0.0610 | 0.6847 | 61.12 |
| Gradient Only | 60 | 26 | 657 | 98 | 0.0381 | 0.2097 | 0.0644 | 0.6910 | 64.35 |
| Texture Only | 60 | 26 | 186 | 98 | 0.1226 | 0.2097 | 0.1548 | 0.6838 | 48.69 |
| Full PGRG | 60 | 21 | 206 | 103 | 0.0925 | 0.1694 | 0.1197 | 0.7300 | 45.76 |

### Best Configuration per Metric

- **Highest Precision**: Texture Only (0.1226)
- **Highest Recall**: Gradient Only (0.2097)
- **Highest F1 Score**: Texture Only (0.1548)
- **Highest Mean IoU**: Full PGRG (0.7300)

---

## 5. Feature Analysis

### Single-Feature Ranking (by F1 Score)

1. **Texture Only**: F1 = 0.1548
2. **Gradient Only**: F1 = 0.0644
3. **Deviation Only**: F1 = 0.0610

The Full PGRG combination achieved a lower F1 (0.1197) than
the best single-feature variant Texture Only (0.1548),
a decrease of 22.7%.

---

## 6. Limitations

1. **Sample size**: The ablation study uses 60 images (10 per class),
   not the full 1,800-image dataset. Results may differ on the full set.
2. **Box-level evaluation**: Metrics are computed from bounding-box
   overlap, not pixel-level segmentation accuracy.
3. **Single weight configuration per variant**: Each single-feature
   variant uses weight = 1.0. Alternative weight magnitudes were not tested.
4. **Runtime variability**: Runtime measurements are single-run per
   image; they may vary with system load.
5. **No hyperparameter tuning**: The seed threshold, similarity threshold,
   and minimum area were held constant across all variants. Performance
   of individual features might improve with tuned parameters.

---

## 7. Conclusion

Among single-feature variants, **Texture Only** achieved the highest
F1 score (0.1548), suggesting it is the most individually
informative feature for defect detection with the PGRG algorithm.
However, the Full PGRG combination (F1 = 0.1197) did not
outperform Texture Only on this sample, suggesting that feature
weighting may benefit from further optimization.

---

## 8. Reproduction

```bash
# From the project root directory:
python experiments/run_ablation_study.py
```

Output files:
- `results/ablation/ablation_results.csv` — per-image results
- `results/ablation/ablation_summary.csv` — aggregate results
- `results/ablation/ablation_comparison.png` — comparison charts
- `results/ablation/ablation_report.md` — this report
