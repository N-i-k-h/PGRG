# PGRG Improvement Report

## Experiment Setup

- **Development subset**: 60 images (10/class from train split)
- **Held-out evaluation**: 360 images (validation split)
- **Evaluation**: Box-level localization, IoU threshold = 0.5
- **Baseline parameters**: seed_threshold=0.45, similarity_threshold=20, min_area=20

## Best Configuration Found

- **seed_threshold**: 0.45
- **similarity_threshold**: 25
- **min_area**: 100
- **merge_gap**: 11
- **weights**: equal

## Held-Out Validation Results

### Overall Comparison

| Method | TP | FP | FN | Precision | Recall | F1 | Mean IoU | Runtime (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Improved PGRG | 110 | 502 | 744 | 0.1797 | 0.1288 | 0.1501 | 0.6980 | 180.87 |
| Original PGRG | 108 | 1792 | 746 | 0.0568 | 0.1265 | 0.0784 | 0.6987 | 135.54 |
| Standard Region Growing | 79 | 8419 | 775 | 0.0093 | 0.0925 | 0.0169 | 0.7113 | 95.22 |
| Otsu + Morphology | 110 | 7855 | 744 | 0.0138 | 0.1288 | 0.0249 | 0.6982 | 0.19 |
| Watershed | 50 | 2470 | 804 | 0.0198 | 0.0585 | 0.0296 | 0.7021 | 1.55 |

### Improvement Deltas (Improved vs Original PGRG)

| Metric | Original | Improved | Delta |
|---|---:|---:|---:|
| Precision | 0.0568 | 0.1797 | +0.1229 |
| Recall | 0.1265 | 0.1288 | +0.0023 |
| F1 | 0.0784 | 0.1501 | +0.0716 |
| Mean IoU | 0.6987 | 0.6980 | -0.0007 |
| FP Count | 1792 | 502 | -1290 |
| Runtime (ms) | 135.54 | 180.87 | +45.33 |

### Class-Wise F1 Scores

| Class | Improved PGRG | Original PGRG | Standard Region Growing | Otsu + Morphology | Watershed |
|---|---:|---:|---:|---:|---:|
| crazing | 0.0154 | 0.0179 | 0.0012 | 0.0040 | 0.0000 |
| inclusion | 0.0611 | 0.0537 | 0.0000 | 0.0000 | 0.0000 |
| patches | 0.1655 | 0.1680 | 0.0000 | 0.0025 | 0.0000 |
| pitted_surface | 0.3333 | 0.1029 | 0.0656 | 0.0650 | 0.0148 |
| rolled-in_scale | 0.0090 | 0.0216 | 0.0012 | 0.0024 | 0.0151 |
| scratches | 0.3217 | 0.1303 | 0.0948 | 0.1804 | 0.2571 |

## Methodology

1. **Phase 1**: 1D parameter sweeps (min_area, seed_threshold, similarity_threshold)
   on 60-image development subset. Best values from each sweep combined.
2. **Phase 2**: Morphological closing tested with gap sizes 0, 3, 5, 7, 9, 11.
3. **Phase 3**: Priority-map weight ablation (10 configurations).
4. **Phase 4**: Best configuration evaluated on held-out 360-image validation set.

## Limitations

- Development subset (60 images) is small; results may vary with different subsets.
- Parameter search is 1D sweeps + top-2 combinations, not exhaustive grid search.
- Morphological closing may merge unrelated nearby defects.
- Evaluation is box-level localization, NOT pixel-level segmentation accuracy.
- Original 1,800-image benchmark results are preserved unchanged.

## Reproduction Commands

```bash
# Full pipeline (all phases):
python experiments/run_improvement_experiments.py --phase all

# Individual phases:
python experiments/run_improvement_experiments.py --phase 1  # Parameter search
python experiments/run_improvement_experiments.py --phase 2  # Region merging
python experiments/run_improvement_experiments.py --phase 3  # Ablation
python experiments/run_improvement_experiments.py --phase 4  # Final evaluation
```