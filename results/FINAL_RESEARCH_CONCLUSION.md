# Final Research Conclusion

**Title**: *Priority-Guided Region Growing for Fast Surface Defect Detection in Steel Strip Images*  
**Dataset**: NEU Surface Defect Database (NEU-DET) — 1,800 images across 6 defect classes  
**Evaluation Protocol**: Box-Level Localization Evaluation (IoU threshold = 0.5)  

---

## 1. Summary of the Proposed Method

The Priority-Guided Region Growing (PGRG) framework is designed as a lightweight, non-deep-learning computer vision algorithm for automated steel surface defect detection. Operating without training overhead or specialized hardware acceleration, PGRG structures defect detection into an eight-stage sequential pipeline:

1. **Preprocessing**: Grayscale conversion, Gaussian spatial smoothing ($5 \times 5$ kernel), and min-max intensity normalization to the range $[0, 255]$ to suppress high-frequency background sensor noise.
2. **Priority Feature Extraction**: Extraction of three complementary surface characteristics per pixel:
   - Local intensity deviation relative to a $7 \times 7$ window mean,
   - Sobel gradient magnitude capturing edge boundaries,
   - Local texture variance capturing textural non-uniformity.
3. **Priority Map Formulation**: Weighted linear combination and min-max normalization into a continuous priority surface $P(x, y) \in [0, 1]$ representing defect saliency.
4. **Seed Selection**: Automatic candidate seed initialization by thresholding the priority map at $T_{\text{seed}} = 0.45$.
5. **Max-Heap Priority Queue**: Priority ordering of seed and candidate pixels using a binary max-heap (implemented via negative priority weighting), ensuring the most salient defect candidates are expanded first.
6. **Priority-Guided Region Growing**: 8-neighbour pixel expansion governed by combined priority ordering and an intensity similarity constraint ($|I(q) - \mu_{\text{region}}| \le 20$).
7. **Union-Find Disjoint Set Management**: Disjoint Set Union (DSU) tracking with path compression and union-by-rank to maintain topological region connectivity with near-linear time complexity $\mathcal{O}(\alpha(N))$.
8. **Region Filtering**: Morphological connected component area filtering with a minimum area threshold ($\text{min\_area} = 20$ pixels) to discard isolated noise artifacts.

---

## 2. Experimental Findings

The full 1,800-image NEU-DET benchmark (300 images per class across 6 defect classes; 7,200 individual method executions) evaluated PGRG against three established classical baselines: Standard Region Growing, Otsu Thresholding + Morphology, and Marker-Controlled Watershed.

The measured overall performance metrics on the full dataset are as follows:

| Method | Precision | Recall | F1 Score | Mean Matched IoU | Runtime (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PGRG** | **0.0646** | **0.1163** | **0.0831** | 0.6789 | 70.86 |
| **Standard Region Growing** | 0.0109 | 0.1120 | 0.0199 | **0.7159** | 49.24 |
| **Otsu + Morphology** | 0.0127 | 0.1153 | 0.0228 | 0.7073 | **0.14** |
| **Watershed** | 0.0207 | 0.0511 | 0.0295 | 0.6894 | 0.98 |

---

## 3. Main Contribution

On the full 1,800-image NEU-DET benchmark, the primary contributions and empirical findings are:

1. **Highest Overall Localization Accuracy**: PGRG achieved the highest **Precision** ($0.0646$), the highest **Recall** ($0.1163$), and the highest **F1 Score** ($0.0831$) among all four evaluated algorithms.
   - PGRG improved overall F1 score by **+317.59%** over Standard Region Growing ($0.0831$ vs $0.0199$), by **+264.47%** over Otsu + Morphology ($0.0831$ vs $0.0228$), and by **+181.69%** over Watershed ($0.0831$ vs $0.0295$).
2. **Tradeoffs Identified**: While PGRG led in Precision, Recall, and F1 score:
   - **Standard Region Growing** achieved the highest Mean Matched-Box IoU ($0.7159$ vs PGRG $0.6789$), indicating that for true-positive detections, standard intensity queue expansion produced bounding boxes that slightly more closely adhered to the ground-truth box boundaries.
   - **Otsu + Morphology** achieved the lowest computational runtime ($0.14$ ms per image), operating several hundred times faster than the iterative region-growing algorithms.

---

## 4. False Positive Suppression

A central hypothesis of priority guidance is the suppression of false defect candidates caused by non-defect surface textures, roll marks, and illumination gradients. The experimental results validate this hypothesis on the benchmark dataset:

- **PGRG False Positives**: **7,049** FP detections across 1,800 images.
- **Standard Region Growing False Positives**: **42,542** FP detections across 1,800 images.
- **Otsu + Morphology False Positives**: **37,671** FP detections across 1,800 images.
- **Watershed False Positives**: **10,116** FP detections across 1,800 images.

### Measured Reductions Achieved by PGRG:
- **83.43% reduction** in false positives compared to Standard Region Growing (a net reduction of 35,493 spurious boxes).
- **81.29% reduction** in false positives compared to Otsu + Morphology (a net reduction of 30,622 spurious boxes).
- **30.32% reduction** in false positives compared to Watershed (a net reduction of 3,067 spurious boxes).

*Note: These reductions represent empirical findings under the evaluated NEU-DET dataset configuration and experimental protocol, and are not claimed as universal across all unconstrained industrial imaging settings.*

---

## 5. Class-Wise Findings

Performance varied substantially across defect morphologies, highlighting the structural strengths and limitations of priority guidance:

### Class-Wise F1 Scores:
- **crazing**: **PGRG = 0.0163** (highest) | Standard RG = 0.0013 | Otsu = 0.0027 | Watershed = 0.0004
- **inclusion**: **PGRG = 0.0971** (highest) | Standard RG = 0.0026 | Otsu = 0.0010 | Watershed = 0.0039
- **patches**: **PGRG = 0.1000** (highest) | Standard RG = 0.0028 | Otsu = 0.0025 | Watershed = 0.0000
- **pitted_surface**: **PGRG = 0.1097** (highest) | Standard RG = 0.0923 | Otsu = 0.0782 | Watershed = 0.0140
- **rolled-in_scale**: **PGRG = 0.0212** (highest) | Standard RG = 0.0014 | Otsu = 0.0020 | Watershed = 0.0086
- **scratches**: PGRG = 0.1783 | Standard RG = 0.1235 | Otsu = 0.1595 | **Watershed = 0.2524** (highest)

### Summary of Class Outcomes:
- PGRG achieved the highest F1 score in **5 out of 6** classes (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`).
- Watershed achieved a higher F1 score on **`scratches`** ($0.2524$ vs PGRG $0.1783$), benefiting from strong edge contrast along linear scratch trajectories.
- **Strongest Defect Classes for PGRG**:
  1. `scratches` ($F_1 = 0.1783$, Precision = $0.1610$, Recall = $0.1997$)
  2. `pitted_surface` ($F_1 = 0.1097$, Precision = $0.0662$, Recall = $0.3211$)
  3. `patches` ($F_1 = 0.1000$, Precision = $0.0992$, Recall = $0.1008$)
  4. `inclusion` ($F_1 = 0.0971$, Precision = $0.1048$, Recall = $0.0905$)

---

## 6. Computational Performance

Experimental runtime measurements reveal the computational profile of the four algorithms on $200 \times 200$ images:

- **PGRG**: **70.86 ms** per image (approx. 14.1 frames per second).
- **Standard Region Growing**: **49.24 ms** per image (approx. 20.3 frames per second).
- **Watershed**: **0.98 ms** per image (approx. 1,020 frames per second).
- **Otsu + Morphology**: **0.14 ms** per image (approx. 7,140 frames per second).

PGRG is **NOT** the fastest method evaluated. It is slower than Standard Region Growing (+43.91% execution time), Watershed ($72.3\times$ slower), and Otsu + Morphology ($506.1\times$ slower). This additional computational overhead is an observed experimental limitation stemming from multi-feature gradient and variance calculations combined with binary heap $\mathcal{O}(\log K)$ insertion and extraction costs.

---

## 7. Evaluation Limitation

An essential methodological limitation must be explicitly recorded:

> **The evaluation protocol employed is BOX-LEVEL LOCALIZATION EVALUATION, not pixel-level segmentation evaluation.**

The NEU-DET benchmark provides PASCAL VOC bounding-box annotations (`<bndbox>` coordinates: `xmin`, `ymin`, `xmax`, `ymax`). Ground-truth pixel masks or boundary polygons are not available in the dataset (`<segmented>0</segmented>` across all XML files). Consequently:
- **Precision** is box-level precision (fraction of predicted bounding boxes matching a ground-truth box with IoU $\ge 0.5$).
- **Recall** is box-level recall (fraction of ground-truth boxes matched by a predicted box with IoU $\ge 0.5$).
- **F1 Score** is the harmonic mean of box-level precision and recall.
- **IoU** represents the intersection-over-union between matched rectangular bounding boxes.

These results must **not** be construed as pixel-level segmentation accuracy.

---

## 8. Limitations

Based strictly on the measured empirical data, the primary limitations of PGRG include:

1. **Higher Runtime Overhead**: At 70.86 ms per image, PGRG is the slowest among all evaluated methods, requiring further optimization for ultra-high-speed rolling mill inspection lines exceeding 30 fps.
2. **Lower Mean Matched IoU than Standard Region Growing**: When true positives are detected, PGRG achieved a Mean Matched IoU of $0.6789$, compared to $0.7159$ for Standard Region Growing and $0.7073$ for Otsu + Morphology.
3. **Suboptimal Scratch Localization Relative to Watershed**: Watershed outperformed PGRG on the `scratches` class ($F_1 = 0.2524$ vs $0.1783$), highlighting that distance-transform watersheds excel on sharp linear grooves.
4. **Low Absolute F1 Scores on Diffuse Classes**: Absolute F1 scores remain low on micro-crack and scale patterns such as `crazing` ($F_1 = 0.0163$) and `rolled-in_scale` ($F_1 = 0.0212$), where low contrast and complex steel strip background noise challenge classical hand-crafted feature extractors.

---

## 9. Final Conclusion

On the 1,800-image NEU-DET benchmark, the Priority-Guided Region Growing (PGRG) algorithm demonstrated a clear, measured advantage in localization precision, recall, and overall F1 score over conventional Standard Region Growing, Otsu thresholding with morphology, and marker-controlled watershed segmentation. By coupling multi-feature saliency estimation with heap-ordered expansion, PGRG suppressed false positive detections by over 81% relative to standard region growing and Otsu thresholding, achieving the highest F1 score in five out of six defect classes. 

However, this detection accuracy improvement comes with clear engineering tradeoffs: PGRG incurs the highest computational latency among evaluated methods (70.86 ms per image), exhibits slightly lower bounding-box overlap quality on matched targets than standard region growing (0.6789 vs 0.7159 IoU), and is outperformed by Watershed on linear scratch defects. Within its operational envelope, PGRG provides an effective classical baseline that bridges the gap between crude intensity thresholding and heavy deep learning models for steel surface defect detection.

---

## 10. Future Work

Directions for future research suggested by these findings include:

1. **Feature Computation Optimization**: Accelerating local variance and Sobel filtering using separable integral images, SIMD vectorization, or OpenCL/CUDA primitives to bring runtime within real-time limits (< 10 ms).
2. **Heap Overhead Reduction**: Investigating quantized bucket queues or hierarchical radices to replace the dynamic binary heap and reduce $\mathcal{O}(\log K)$ queue manipulation costs.
3. **Recall and Boundary Sensitivity Enhancement**: Implementing adaptive, class-conditioned similarity thresholds to elevate recall in faint defect classes such as `crazing` and `rolled-in_scale`.
4. **Hybrid Scratch Segmentation**: Exploring hybrid watershed-priority integration specifically tailored for linear defect geometries.
5. **Pixel-Level Ground Truth Acquisition**: Establishing a verified pixel-level steel defect benchmark to evaluate true morphological boundary delineation beyond bounding box approximations.
6. **Parameter Sensitivity Analysis**: Systematically exploring the hyperparameter response surface (e.g., $T_{\text{seed}}$, similarity threshold $\Delta I$, feature weighting coefficients $w_1, w_2, w_3$) to establish optimal operational frontiers.
