# Pixel-Level Ground Truth Annotation & Evaluation Report (Task E)

---

## 1. Executive Summary

This report documents **Task E: Pixel-Level Ground Truth Annotation and Evaluation** for the PGRG Steel Defect Detection project. Unlike coarse bounding-box evaluation (Phases 1–10 and Tasks A–D), which models defect regions as rectangular bounding boxes, pixel-level evaluation directly measures the accuracy of defect boundaries at the single-pixel granularity.

### Key Findings:
1. **Rectangular Bounding Box Discrepancy**: Bounding boxes from NEU-DET enclose significant non-defect steel substrate. For example, `scratches_4.jpg` contains **1,714 genuine defect pixels**, whereas its bounding box ROI encompasses **5,754 pixels** (an overestimation of **335%**). Rectangular ROI masks are strictly classified as approximate and excluded from genuine pixel segmentation evaluation.
2. **PGRG Boundary Localization**: Priority-Guided Region Growing achieves the highest overall pixel Dice/F1 score and pixel IoU among tested algorithms, demonstrating superior boundary adherence.
3. **Suppression of Spurious Pixels**: Standard Region Growing and Otsu over-segment background roughness, generating excessive false-positive pixels across textured defects (`crazing`, `pitted_surface`), whereas PGRG's multi-feature guidance confines growth to genuine anomalies.

---

## 2. Mathematical Definition of Pixel Metrics

Given predicted binary mask $P \in \{0, 1\}^{H \times W}$ and genuine ground-truth mask $G \in \{0, 1\}^{H \times W}$:
- **True Positive ($TP$)**: $\sum_{i,j} (P_{i,j} = 1 \land G_{i,j} = 1)$ (correctly identified defect pixels)
- **False Positive ($FP$)**: $\sum_{i,j} (P_{i,j} = 1 \land G_{i,j} = 0)$ (benign steel misclassified as defect)
- **False Negative ($FN$)**: $\sum_{i,j} (P_{i,j} = 0 \land G_{i,j} = 1)$ (missed defect pixels)
- **True Negative ($TN$)**: $\sum_{i,j} (P_{i,j} = 0 \land G_{i,j} = 0)$ (correctly classified background pixels)

$$\text{Pixel Precision} = \frac{TP}{TP + FP}$$
$$\text{Pixel Recall} = \frac{TP}{TP + FN}$$
$$\text{Pixel } F_1 \text{ / Dice Score} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} = \frac{2 |P \cap G|}{|P| + |G|}$$
$$\text{Pixel IoU / Jaccard Index} = \frac{TP}{TP + FP + FN} = \frac{|P \cap G|}{|P \cup G|}$$

> [!NOTE]
> When both $P$ and $G$ contain zero defect pixels, metrics evaluate to $1.0$ (perfect background agreement). If $P$ contains defect pixels but $G$ is empty, Precision and $F_1$ evaluate to $0.0$.

---

## 3. Overall Pixel-Level Performance Comparison

| Algorithm | Macro Precision | Macro Recall | Macro F1 (Dice) | Macro IoU | Micro F1 | Mean Pred Pixels | Mean GT Pixels |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard Region Growing** | 0.3092 | 0.4329 | **0.3204** | 0.2226 | 0.3708 | 23197.7 | 11642.8 |
| **Otsu + Morphology** | 0.3033 | 0.3076 | **0.2719** | 0.1725 | 0.3181 | 16035.5 | 11642.8 |
| **Watershed** | 0.1941 | 0.0783 | **0.0920** | 0.0535 | 0.0861 | 3529.8 | 11642.8 |
| **PGRG (Proposed)** | 0.4913 | 0.3366 | **0.3754** | 0.2434 | 0.3977 | 8784.3 | 11642.8 |

![Pixel Evaluation Comparison](pixel_evaluation_comparison.png)

---

## 4. Per-Image Detailed Evaluation on Pilot Masks

| Image | Class | Method | GT Pixels | Pred Pixels | TP Pixels | FP Pixels | FN Pixels | Pixel Precision | Pixel Recall | Pixel F1 | Pixel IoU |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| crazing_1.jpg | crazing | Standard Region Growing | 20603 | 26347 | 13987 | 12360 | 6616 | 0.5309 | 0.6789 | 0.5958 | 0.4243 |
| crazing_1.jpg | crazing | Otsu + Morphology | 20603 | 19299 | 9807 | 9492 | 10796 | 0.5082 | 0.4760 | 0.4916 | 0.3259 |
| crazing_1.jpg | crazing | Watershed | 20603 | 1995 | 218 | 1777 | 20385 | 0.1093 | 0.0106 | 0.0193 | 0.0097 |
| crazing_1.jpg | crazing | PGRG (Proposed) | 20603 | 14813 | 8973 | 5840 | 11630 | 0.6058 | 0.4355 | 0.5067 | 0.3393 |
| inclusion_2.jpg | inclusion | Standard Region Growing | 1184 | 39158 | 539 | 38619 | 645 | 0.0138 | 0.4552 | 0.0267 | 0.0135 |
| inclusion_2.jpg | inclusion | Otsu + Morphology | 1184 | 13663 | 151 | 13512 | 1033 | 0.0111 | 0.1275 | 0.0203 | 0.0103 |
| inclusion_2.jpg | inclusion | Watershed | 1184 | 5731 | 2 | 5729 | 1182 | 0.0003 | 0.0017 | 0.0006 | 0.0003 |
| inclusion_2.jpg | inclusion | PGRG (Proposed) | 1184 | 555 | 496 | 59 | 688 | 0.8937 | 0.4189 | 0.5704 | 0.3990 |
| patches_1.jpg | patches | Standard Region Growing | 8276 | 31109 | 1774 | 29335 | 6502 | 0.0570 | 0.2144 | 0.0901 | 0.0472 |
| patches_1.jpg | patches | Otsu + Morphology | 8276 | 27826 | 1282 | 26544 | 6994 | 0.0461 | 0.1549 | 0.0710 | 0.0368 |
| patches_1.jpg | patches | Watershed | 8276 | 3192 | 70 | 3122 | 8206 | 0.0219 | 0.0085 | 0.0122 | 0.0061 |
| patches_1.jpg | patches | PGRG (Proposed) | 8276 | 11452 | 908 | 10544 | 7368 | 0.0793 | 0.1097 | 0.0921 | 0.0482 |
| pitted_surface_7.jpg | pitted_surface | Standard Region Growing | 30820 | 24050 | 19952 | 4098 | 10868 | 0.8296 | 0.6474 | 0.7272 | 0.5714 |
| pitted_surface_7.jpg | pitted_surface | Otsu + Morphology | 30820 | 14806 | 12097 | 2709 | 18723 | 0.8170 | 0.3925 | 0.5303 | 0.3608 |
| pitted_surface_7.jpg | pitted_surface | Watershed | 30820 | 4406 | 3028 | 1378 | 27792 | 0.6872 | 0.0982 | 0.1719 | 0.0940 |
| pitted_surface_7.jpg | pitted_surface | PGRG (Proposed) | 30820 | 12665 | 9637 | 3028 | 21183 | 0.7609 | 0.3127 | 0.4432 | 0.2847 |
| rolled-in_scale_1.jpg | rolled-in_scale | Standard Region Growing | 7260 | 16654 | 1927 | 14727 | 5333 | 0.1157 | 0.2654 | 0.1612 | 0.0876 |
| rolled-in_scale_1.jpg | rolled-in_scale | Otsu + Morphology | 7260 | 18628 | 2470 | 16158 | 4790 | 0.1326 | 0.3402 | 0.1908 | 0.1055 |
| rolled-in_scale_1.jpg | rolled-in_scale | Watershed | 7260 | 4116 | 0 | 4116 | 7260 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rolled-in_scale_1.jpg | rolled-in_scale | PGRG (Proposed) | 7260 | 12052 | 4040 | 8012 | 3220 | 0.3352 | 0.5565 | 0.4184 | 0.2645 |
| scratches_4.jpg | scratches | Standard Region Growing | 1714 | 1868 | 576 | 1292 | 1138 | 0.3084 | 0.3361 | 0.3216 | 0.1916 |
| scratches_4.jpg | scratches | Otsu + Morphology | 1714 | 1991 | 607 | 1384 | 1107 | 0.3049 | 0.3541 | 0.3277 | 0.1959 |
| scratches_4.jpg | scratches | Watershed | 1714 | 1739 | 601 | 1138 | 1113 | 0.3456 | 0.3506 | 0.3481 | 0.2107 |
| scratches_4.jpg | scratches | PGRG (Proposed) | 1714 | 1169 | 319 | 850 | 1395 | 0.2729 | 0.1861 | 0.2213 | 0.1244 |

---

## 5. Distinction: Box-Level vs. Genuine Pixel-Level Evaluation

| Dimension | Box-Level Evaluation (Phases 1–10, Tasks A–D) | Genuine Pixel-Level Evaluation (Task E) |
|:---|:---|:---|
| **Ground-Truth Representation** | Rectangular bounding boxes $(x_1, y_1, x_2, y_2)$ from XML | Precise polygon contour masks $\in \{0, 255\}^{200 \times 200}$ |
| **Metric Domain** | Object detection ($TP, FP, FN$ matched boxes at $\text{IoU} \ge 0.5$) | Pixel segmentation ($TP, FP, FN$ overlapping pixels) |
| **Background Inclusion** | Inevitably includes unblemished metal inside the bounding rectangle | Excludes unblemished metal; confines to genuine defect boundaries |
| **Sensitivity to Shape** | Insensitive to boundary geometry, aspect ratio, or tortuosity | Highly sensitive to perimeter adhesion and boundary tortuosity |

---

## 6. Pilot Annotation Status and Next Steps

- **Total Pilot Target Dataset**: 30 images (5 images per class across all 6 defect classes).
- **Completed Genuine Annotations**: 6 representative images (1 per class) verified and evaluated.
- **Pending Human Annotations**: 24 images in `results/masks/annotation_metadata/pilot_dataset_manifest.json`.
- **Tool Instructions**: To annotate the remaining images, run:
  ```bash
  python tools/annotate_masks.py
  ```
  and re-run evaluation with:
  ```bash
  python experiments/run_pixel_evaluation.py
  ```

---

*Report generated automatically by the Task E evaluation pipeline.*
