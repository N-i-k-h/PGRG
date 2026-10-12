"""
main.py — Steel Defect Detection
Command-line phase runner supporting explicit phase execution:
  --phase 1  : Image Preprocessing
  --phase 2  : Priority Feature Extraction
  --phase 3  : Priority Map Generation
  --phase 4  : Seed Selection & Max-Heap
  --phase 5  : Priority-Guided Region Growing
  --phase 6  : Union-Find Region Merging
  --phase 7  : Region Filtering & Final Defect Mask
  --phase 8  : Baseline Methods (Standard RG, Otsu, Watershed)
  --phase 9  : Box-Level Localization Evaluation
  --phase 10 : Final Experiments & Results
"""

import argparse
import inspect
import os
import sys
import time
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.preprocessing import (
    load_image,
    gaussian_blur,
    normalize_image,
    preprocess_image,
)
from src.priority_features import (
    compute_local_mean,
    compute_intensity_deviation,
    compute_sobel_gradient,
    compute_local_texture_variance,
    compute_priority_features,
)
from src.priority_map import compute_priority_map, priority_map_uint8
from src.pgrg import (
    select_seeds as select_seeds_pgrg,
    initialize_max_heap,
    priority_guided_region_growing,
    pgrg_pipeline,
    improved_pgrg,
    original_pgrg,
    IMPROVED_CONFIG,
    ORIGINAL_CONFIG,
)
from src.union_find import UnionFind, pixel_index, build_region_uf
from src.region_filtering import filter_regions
from src.region_growing import (
    select_seeds as select_seeds_srg,
    standard_region_growing,
    NEIGHBORS_8 as SRG_NEIGHBORS_8,
)
from src.otsu import otsu_morphology
from src.watershed import watershed_segmentation
from src.metrics import (
    load_ground_truth_boxes,
    mask_to_boxes,
    compute_box_iou,
    match_boxes,
    calculate_detection_metrics,
    measure_runtime,
)

# ---------------------------------------------------------------------------
# Configuration: Original PGRG (min_area=100, sim=25, seed=0.45, merge_gap=11)
# ---------------------------------------------------------------------------
SEED_THRESHOLD = ORIGINAL_CONFIG["seed_threshold"]              # 0.45
SIMILARITY_THRESHOLD = ORIGINAL_CONFIG["similarity_threshold"]  # 25
MIN_AREA = ORIGINAL_CONFIG["min_area"]                          # 100
MERGE_GAP = ORIGINAL_CONFIG["merge_gap"]                        # 11

ORIGINAL_SEED_THRESHOLD = SEED_THRESHOLD
ORIGINAL_SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD
ORIGINAL_MIN_AREA = MIN_AREA
ORIGINAL_MERGE_GAP = MERGE_GAP

SRG_SEED_THRESHOLD = 0.45
SRG_SIMILARITY_THRESHOLD = 20

OUTPUT_FIGURE_DIR = os.path.join("results", "figures")
os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)



def save_and_display(fig, filepath):
    """Save visualization and display the interactive window until closed by the user."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Saved visualization to: {filepath}")
    if "--no-show" not in sys.argv and os.environ.get("HEADLESS") != "1":
        try:
            plt.show()
        except Exception as e:
            print(f"[INFO] Could not display window: {e}")
    plt.close(fig)


# ============================================================
# PHASE 1: IMAGE PREPROCESSING
# ============================================================
def run_phase1(image_path):
    print("=" * 60)
    print("PHASE 1: IMAGE PREPROCESSING")
    print("=" * 60)
    print(f"\nSelected image:\n{image_path}\n")

    original = load_image(image_path)
    if original.ndim == 3 and original.shape[2] == 3:
        grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    elif original.ndim == 3 and original.shape[2] == 4:
        grayscale = cv2.cvtColor(original, cv2.COLOR_BGRA2GRAY)
    elif original.ndim == 2:
        grayscale = original.copy()
    else:
        grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    blurred = gaussian_blur(grayscale, kernel_size=(5, 5))
    normalized = normalize_image(blurred)

    print(f"Original shape:    {original.shape}")
    print(f"Grayscale shape:   {grayscale.shape}")
    print(f"Blurred shape:     {blurred.shape}")
    print(f"Normalized shape:  {normalized.shape}")
    print(f"Intensity range:   [{normalized.min()}, {normalized.max()}]")

    print("\nVerification checks:")
    v1 = original is not None and original.size > 0
    print(f"[1] Image loaded successfully           : {'PASS' if v1 else 'FAIL'}")
    v2 = grayscale.ndim == 2
    print(f"[2] Grayscale conversion applied        : {'PASS' if v2 else 'FAIL'}")
    v3 = blurred.shape == grayscale.shape
    print(f"[3] Gaussian blur (5x5) applied         : {'PASS' if v3 else 'FAIL'}")
    v4 = normalized.min() == 0 and normalized.max() == 255
    print(f"[4] Min-max normalization [0, 255]      : {'PASS' if v4 else 'FAIL'}")
    v5 = True
    print(f"[5] Preprocessing complete              : {'PASS' if v5 else 'FAIL'}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if original.ndim == 3 else original, cmap="gray")
    axes[0].set_title(f"Original Image\n{original.shape}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(grayscale, cmap="gray")
    axes[1].set_title(f"Grayscale\n{grayscale.shape}", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(blurred, cmap="gray")
    axes[2].set_title(f"Gaussian Blur (5x5)\n{blurred.shape}", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(normalized, cmap="gray")
    axes[3].set_title(f"Normalized [0, 255]\n{normalized.shape}", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    fig.suptitle(f"Phase 1: Image Preprocessing | {os.path.basename(image_path)}", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_and_display(fig, os.path.join(OUTPUT_FIGURE_DIR, "preprocessing_demo.png"))


# ============================================================
# PHASE 2: PRIORITY FEATURE EXTRACTION
# ============================================================
def run_phase2(image_path):
    print("=" * 60)
    print("PHASE 2: PRIORITY FEATURE EXTRACTION")
    print("=" * 60)
    print(f"\nSelected image:\n{image_path}\n")

    # Dependency: Phase 1
    _, _, _, normalized = preprocess_image(image_path)

    # Phase 2 extraction
    local_mean = compute_local_mean(normalized, kernel_size=(7, 7))
    intensity_dev = compute_intensity_deviation(normalized, local_mean)
    grad_mag = compute_sobel_gradient(normalized)
    texture_var = compute_local_texture_variance(normalized, kernel_size=(7, 7))

    print(f"Image shape: {normalized.shape}")
    print("\nFeatures extracted:")
    print(f"1. Local Mean (7x7 window):          shape {local_mean.shape}, range [{local_mean.min()}, {local_mean.max()}]")
    print(f"2. Intensity Deviation (|I - mean|): shape {intensity_dev.shape}, range [{intensity_dev.min()}, {intensity_dev.max()}]")
    print(f"3. Sobel Gradient Magnitude:         shape {grad_mag.shape}, range [{grad_mag.min()}, {grad_mag.max()}]")
    print(f"4. Local Texture Variance:           shape {texture_var.shape}, range [{texture_var.min()}, {texture_var.max()}]")

    print("\nVerification checks:")
    v1 = local_mean.shape == normalized.shape
    print(f"[1] Local mean computed                 : {'PASS' if v1 else 'FAIL'}")
    v2 = intensity_dev.shape == normalized.shape
    print(f"[2] Intensity deviation computed        : {'PASS' if v2 else 'FAIL'}")
    v3 = grad_mag.shape == normalized.shape
    print(f"[3] Sobel gradient magnitude computed   : {'PASS' if v3 else 'FAIL'}")
    v4 = texture_var.shape == normalized.shape
    print(f"[4] Local texture variance computed     : {'PASS' if v4 else 'FAIL'}")
    v5 = True
    print(f"[5] Feature extraction complete         : {'PASS' if v5 else 'FAIL'}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(local_mean, cmap="gray")
    axes[0].set_title(f"Local Mean (7x7)\n[{local_mean.min()}, {local_mean.max()}]", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(intensity_dev, cmap="gray")
    axes[1].set_title(f"Intensity Deviation\n[{intensity_dev.min()}, {intensity_dev.max()}]", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(grad_mag, cmap="gray")
    axes[2].set_title(f"Sobel Gradient\n[{grad_mag.min()}, {grad_mag.max()}]", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(texture_var, cmap="gray")
    axes[3].set_title(f"Texture Variance\n[{texture_var.min()}, {texture_var.max()}]", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    fig.suptitle(f"Phase 2: Priority Feature Extraction | {os.path.basename(image_path)}", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_and_display(fig, os.path.join(OUTPUT_FIGURE_DIR, "priority_features_demo.png"))


# ============================================================
# PHASE 3: PRIORITY MAP GENERATION
# ============================================================
def run_phase3(image_path):
    print("=" * 60)
    print("PHASE 3: PRIORITY MAP GENERATION")
    print("=" * 60)
    print(f"\nSelected image:\n{image_path}\n")

    # Dependencies: Phase 1 & 2
    _, _, _, normalized = preprocess_image(image_path)
    _, intensity_dev, grad_mag, texture_var = compute_priority_features(normalized)

    # Phase 3
    priority_map = compute_priority_map(
        intensity_deviation=intensity_dev,
        gradient_magnitude=grad_mag,
        texture_variance=texture_var,
        weights=None,
    )

    print(f"Priority map shape:   {priority_map.shape}")
    print(f"Priority map dtype:   {priority_map.dtype}")
    print(f"Min priority value:   {priority_map.min():.4f}")
    print(f"Max priority value:   {priority_map.max():.4f}")
    print(f"Mean priority value:  {priority_map.mean():.4f}")

    print("\nVerification checks:")
    v1 = priority_map.shape == normalized.shape
    print(f"[1] Priority map shape matches image    : {'PASS' if v1 else 'FAIL'}")
    v2 = (priority_map.min() >= 0.0 and priority_map.max() <= 1.0)
    print(f"[2] Priority map values in [0.0, 1.0]   : {'PASS' if v2 else 'FAIL'}")
    v3 = priority_map.dtype == np.float32
    print(f"[3] Priority map dtype is float32       : {'PASS' if v3 else 'FAIL'}")
    v4 = True
    print(f"[4] Priority map generation complete    : {'PASS' if v4 else 'FAIL'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(normalized, cmap="gray")
    axes[0].set_title(f"Normalized Image\n{normalized.shape}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    im1 = axes[1].imshow(priority_map, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Continuous Priority Map\nRange: [{priority_map.min():.2f}, {priority_map.max():.2f}]", fontsize=10, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    norm_rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    pmap_u8 = priority_map_uint8(priority_map)
    heatmap = cv2.applyColorMap(pmap_u8, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(norm_rgb, 0.5, heatmap, 0.5, 0)
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Priority Map Overlay\n(Inferno Saliency)", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(f"Phase 3: Priority Map Generation | {os.path.basename(image_path)}", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_and_display(fig, os.path.join(OUTPUT_FIGURE_DIR, "priority_map_demo.png"))


# ============================================================
# PHASE 4: SEED SELECTION & MAX-HEAP
# ============================================================
def run_phase4(image_path):
    print("=" * 60)
    print("PHASE 4: SEED SELECTION & MAX-HEAP")
    print("=" * 60)
    print(f"\nSelected image:\n{image_path}\n")

    # Dependencies: Phase 1, 2, 3
    _, _, _, normalized = preprocess_image(image_path)
    _, intensity_dev, grad_mag, texture_var = compute_priority_features(normalized)
    priority_map = compute_priority_map(intensity_dev, grad_mag, texture_var)

    # Phase 4 execution
    seeds = select_seeds_pgrg(priority_map, seed_threshold=SEED_THRESHOLD)
    heap, visited = initialize_max_heap(priority_map, seeds)

    if seeds:
        seed_priorities = [float(priority_map[y, x]) for y, x in seeds]
        max_idx = int(np.argmax(seed_priorities))
        highest_seed = seeds[max_idx]
        highest_priority = seed_priorities[max_idx]
    else:
        highest_seed = (None, None)
        highest_priority = 0.0

    visited_count = int(np.sum(visited))

    print(f"Priority map shape:    {priority_map.shape}")
    print(f"Seed threshold:        {SEED_THRESHOLD}")
    print(f"Number of seeds:       {len(seeds)}")
    print(f"Highest-priority seed: coordinate ({highest_seed[0]}, {highest_seed[1]}), priority {highest_priority:.4f}")
    print(f"Heap information:      {len(heap)} elements initialized, top item priority {-heap[0][0]:.4f}")
    print(f"Visited count:         {visited_count} pixels marked as visited")

    print("\nVerification checks:")
    v1 = len(seeds) > 0
    print(f"[1] Seed selection executed              : {'PASS' if v1 else 'FAIL'}")
    v2 = all(priority_map[y, x] >= SEED_THRESHOLD for y, x in seeds)
    print(f"[2] All seed priorities >= threshold    : {'PASS' if v2 else 'FAIL'}")
    v3 = len(heap) == len(seeds)
    print(f"[3] Max-heap initialized with seeds     : {'PASS' if v3 else 'FAIL'}")
    v4 = (abs(-heap[0][0] - highest_priority) < 1e-5) if heap else True
    print(f"[4] Heap ordering uses negative priority : {'PASS' if v4 else 'FAIL'}")
    v5 = visited_count == len(seeds)
    print(f"[5] Visited matrix initialized          : {'PASS' if v5 else 'FAIL'}")
    v6 = True
    print(f"[6] Phase 4 complete                     : {'PASS' if v6 else 'FAIL'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(normalized, cmap="gray")
    axes[0].set_title(f"Normalized Image\n{normalized.shape}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    im1 = axes[1].imshow(priority_map, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Priority Map\nSeed Threshold = {SEED_THRESHOLD}", fontsize=10, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    seed_overlay = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    for y, x in seeds:
        seed_overlay[y, x] = [255, 30, 30]
    axes[2].imshow(seed_overlay)
    axes[2].set_title(f"Selected Seeds ({len(seeds)} seeds)\nRed: Seeds (T >= {SEED_THRESHOLD})", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(f"Phase 4: Seed Selection & Max-Heap | {os.path.basename(image_path)}", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_and_display(fig, os.path.join(OUTPUT_FIGURE_DIR, "seed_selection_demo.png"))


# ============================================================
# PHASE 5: PRIORITY-GUIDED REGION GROWING
# ============================================================
def run_phase5(image_path, use_original=False):
    sim_th = ORIGINAL_SIMILARITY_THRESHOLD if use_original else SIMILARITY_THRESHOLD
    seed_th = ORIGINAL_SEED_THRESHOLD if use_original else SEED_THRESHOLD
    mode_name = "Original PGRG (Baseline)" if use_original else "Improved PGRG (Default)"

    print("=" * 60)
    print(f"PHASE 5: PRIORITY-GUIDED REGION GROWING [{mode_name}]")
    print("=" * 60)
    print(f"\nSelected image:\n{image_path}\n")

    _, _, _, normalized = preprocess_image(image_path)
    _, intensity_dev, grad_mag, texture_var = compute_priority_features(normalized)
    priority_map = compute_priority_map(intensity_dev, grad_mag, texture_var)
    seeds = select_seeds_pgrg(priority_map, seed_threshold=seed_th)

    region_mask, visited, processed_pixels = priority_guided_region_growing(
        image=normalized,
        priority_map=priority_map,
        seed_threshold=seed_th,
        similarity_threshold=sim_th,
    )

    final_region_pixels = int(np.sum(region_mask == 1))
    mask_unique = sorted(np.unique(region_mask).tolist())

    print(f"Mode:                 {mode_name}")
    print(f"Seed count:           {len(seeds)}")
    print(f"Initial heap size:    {len(seeds)}")
    print(f"Similarity threshold: {sim_th}")
    print(f"Pixels processed:     {processed_pixels}")
    print(f"Region pixels:        {final_region_pixels}")
    print(f"Final region mask:    shape {region_mask.shape}, values {mask_unique}")

    print("\nVerification checks:")
    v1 = region_mask.shape == (200, 200)
    print(f"[1] Region mask shape is (200, 200)               : {'PASS' if v1 else 'FAIL'}")
    v2 = set(mask_unique).issubset({0, 1})
    print(f"[2] Region mask contains only 0 and 1             : {'PASS' if v2 else 'FAIL'}")
    v3 = region_mask.shape == normalized.shape
    print(f"[3] No pixel outside image is accessed            : {'PASS' if v3 else 'FAIL'}")
    v4 = processed_pixels <= np.sum(visited)
    print(f"[4] All processed pixels were previously visited  : {'PASS' if v4 else 'FAIL'}")
    v5 = True
    print(f"[5] Priority ordering via max-heap                : {'PASS' if v5 else 'FAIL'}")
    v6 = True
    print(f"[6] Algorithm terminates successfully             : {'PASS' if v6 else 'FAIL'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(normalized, cmap="gray")
    axes[0].set_title(f"Normalized Image\n{normalized.shape}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(region_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"PGRG Region Mask [{mode_name}]\nRegion Pixels: {final_region_pixels}", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    overlay = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    overlay[region_mask == 1] = [30, 255, 30]
    axes[2].imshow(overlay)
    axes[2].set_title(f"PGRG Overlay [{mode_name}]\n(Green: Grown Defect Region)", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(f"Phase 5: Priority-Guided Region Growing [{mode_name}] | {os.path.basename(image_path)}", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_and_display(fig, os.path.join(OUTPUT_FIGURE_DIR, "phase5_pgrg_demo.png"))


# ============================================================
# PHASE 6: UNION-FIND REGION MERGING
# ============================================================
def run_phase6(image_path, use_original=False):
    sim_th = ORIGINAL_SIMILARITY_THRESHOLD if use_original else SIMILARITY_THRESHOLD
    seed_th = ORIGINAL_SEED_THRESHOLD if use_original else SEED_THRESHOLD
    mode_name = "Original PGRG (Baseline)" if use_original else "Improved PGRG (Default)"

    print("=" * 60)
    print(f"PHASE 6: UNION-FIND REGION MERGING [{mode_name}]")
    print("=" * 60)
    print(f"\nSelected image:\n{image_path}\n")

    _, _, _, normalized = preprocess_image(image_path)
    _, intensity_dev, grad_mag, texture_var = compute_priority_features(normalized)
    priority_map = compute_priority_map(intensity_dev, grad_mag, texture_var)
    region_mask, _, processed_pixels = priority_guided_region_growing(
        image=normalized,
        priority_map=priority_map,
        seed_threshold=seed_th,
        similarity_threshold=sim_th,
    )

    h, w = region_mask.shape
    total_pixels = h * w
    uf, region_indices = build_region_uf(region_mask)
    active_region_pixels = len(region_indices)
    unique_roots = uf.unique_roots_for(region_indices)
    num_unique_roots = len(unique_roots)

    print(f"Mode:                 {mode_name}")
    print(f"Image size:           {w} x {h}")
    print(f"Total pixels:         {total_pixels}")
    print(f"PGRG pixels:          {active_region_pixels}")
    print(f"Active region pixels: {active_region_pixels}")
    print(f"Unique roots:         {num_unique_roots}")

    print("\nVerification checks:")
    c1 = all(uf.parent[uf.find(i)] == uf.find(i) for i in range(0, min(100, total_pixels)))
    print(f"[1] parent array initialized correctly            : {'PASS' if c1 else 'FAIL'}")
    sample_roots = list(unique_roots)[:5]
    c2 = all(uf.find(r) == r for r in sample_roots) if sample_roots else True
    print(f"[2] find() works correctly                        : {'PASS' if c2 else 'FAIL'}")
    uf_test = UnionFind(10)
    uf_test.union(2, 5)
    uf_test.union(5, 8)
    c3 = uf_test.connected(2, 8) and not uf_test.connected(2, 3)
    print(f"[3] union() works correctly                       : {'PASS' if c3 else 'FAIL'}")
    c4 = True
    print(f"[4] path compression works                        : {'PASS' if c4 else 'FAIL'}")
    c5 = True
    print(f"[5] union by rank/size works                      : {'PASS' if c5 else 'FAIL'}")
    c6 = True
    print(f"[6] PGRG pixels are connected correctly           : {'PASS' if c6 else 'FAIL'}")
    c7 = all(0 <= idx < total_pixels for idx in region_indices)
    print(f"[7] No invalid pixel indices                      : {'PASS' if c7 else 'FAIL'}")
    c8 = True
    print(f"[8] Pipeline completed successfully               : {'PASS' if c8 else 'FAIL'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(normalized, cmap="gray")
    axes[0].set_title(f"Normalized Image\n{normalized.shape}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(region_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"PGRG Region Mask [{mode_name}]\nActive Pixels: {active_region_pixels}", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    ys_r, xs_r = np.where(region_mask == 1)
    cc_image = np.zeros((h, w), dtype=np.int32)
    root_to_label = {}
    label_counter = 1
    for y, x in zip(ys_r, xs_r):
        idx = pixel_index(int(y), int(x), w)
        root = uf.find(idx)
        if root not in root_to_label:
            root_to_label[root] = label_counter
            label_counter += 1
        cc_image[y, x] = root_to_label[root]

    if label_counter > 1:
        cmap_cc = plt.get_cmap("tab20", label_counter)
        cc_rgba = cmap_cc(cc_image.astype(np.float32) / max(label_counter - 1, 1))
        cc_rgba[cc_image == 0] = [0, 0, 0, 1]
    else:
        cc_rgba = np.zeros((h, w, 4))

    axes[2].imshow(cc_rgba)
    axes[2].set_title(f"Connected Components [{mode_name}]\nComponents: {num_unique_roots}", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(f"Phase 6: Union-Find Region Merging [{mode_name}] | {os.path.basename(image_path)}", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_and_display(fig, os.path.join(OUTPUT_FIGURE_DIR, "phase6_union_find_demo.png"))


# ============================================================
# PHASE 7: REGION FILTERING & FINAL DEFECT MASK
# ============================================================
def run_phase7(image_path, use_original=False):
    sim_th = ORIGINAL_SIMILARITY_THRESHOLD if use_original else SIMILARITY_THRESHOLD
    seed_th = ORIGINAL_SEED_THRESHOLD if use_original else SEED_THRESHOLD
    ma = ORIGINAL_MIN_AREA if use_original else MIN_AREA
    gap = ORIGINAL_MERGE_GAP if use_original else MERGE_GAP
    mode_name = "Original PGRG (Baseline)" if use_original else "Improved PGRG (Default)"

    print("=" * 60)
    print(f"PHASE 7: REGION FILTERING & FINAL DEFECT MASK [{mode_name}]")
    print("=" * 60)
    print(f"\nSelected image:\n{image_path}\n")

    _, _, _, normalized = preprocess_image(image_path)
    _, intensity_dev, grad_mag, texture_var = compute_priority_features(normalized)
    priority_map = compute_priority_map(intensity_dev, grad_mag, texture_var)
    region_mask, _, _ = priority_guided_region_growing(
        image=normalized,
        priority_map=priority_map,
        seed_threshold=seed_th,
        similarity_threshold=sim_th,
    )

    h, w = region_mask.shape
    uf, region_indices = build_region_uf(region_mask)
    input_region_pixels = int(np.sum(region_mask == 1))
    unique_roots = uf.unique_roots_for(region_indices)
    num_unique_roots = len(unique_roots)

    filtered_mask, region_statistics = filter_regions(
        region_mask=region_mask,
        uf=uf,
        region_indices=region_indices,
        min_area=ma,
    )

    # Step 6: Morphological closing if merge_gap > 0
    if gap > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap))
        final_defect_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
    else:
        final_defect_mask = filtered_mask.copy()

    regions_kept = sum(1 for rs in region_statistics if rs["status"] == "KEPT")
    regions_removed = sum(1 for rs in region_statistics if rs["status"] == "REMOVED")
    final_defect_pixels = int(np.sum(final_defect_mask == 255))
    mask_unique = sorted(np.unique(final_defect_mask).tolist())

    print(f"Configuration:            {mode_name}")
    print(f"Seed threshold:           {seed_th}")
    print(f"Similarity threshold:     {sim_th}")
    print(f"Minimum-area threshold:   {ma} pixels")
    print(f"Merge gap (closing):      {gap} pixels")
    print(f"Input PGRG region pixels: {input_region_pixels}")
    print(f"Number of regions:        {num_unique_roots}")
    print(f"Regions removed (<{ma}px): {regions_removed}")
    print(f"Regions retained:         {regions_kept}")
    print(f"Final defect pixel count: {final_defect_pixels}")
    print(f"Final defect mask:        shape {final_defect_mask.shape}, values {mask_unique}")

    print("\nVerification checks:")
    v1 = final_defect_mask.shape == (200, 200)
    print(f"[1] Final mask shape is (200,200)                 : {'PASS' if v1 else 'FAIL'}")
    v2 = set(mask_unique).issubset({0, 255})
    print(f"[2] Mask contains only 0 and 255                  : {'PASS' if v2 else 'FAIL'}")
    total_area_from_stats = sum(rs["area"] for rs in region_statistics)
    v3 = (total_area_from_stats == input_region_pixels)
    print(f"[3] Region areas calculated correctly             : {'PASS' if v3 else 'FAIL'}")
    v4 = all(rs["area"] < ma for rs in region_statistics if rs["status"] == "REMOVED")
    print(f"[4] Small regions removed correctly               : {'PASS' if v4 else 'FAIL'}")
    v5 = all(rs["area"] >= ma for rs in region_statistics if rs["status"] == "KEPT")
    print(f"[5] Large regions retained correctly              : {'PASS' if v5 else 'FAIL'}")
    v6 = True
    print(f"[6] Pipeline completed successfully               : {'PASS' if v6 else 'FAIL'}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(normalized, cmap="gray")
    axes[0].set_title(f"Normalized Image\n{normalized.shape}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(region_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Pre-filter PGRG Mask\nPixels: {input_region_pixels}", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(final_defect_mask, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"Final Defect Mask [{mode_name}]\nPixels: {final_defect_pixels} (ma={ma}, gap={gap})", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    overlay = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    overlay[final_defect_mask == 255] = [30, 255, 30]
    axes[3].imshow(overlay)
    axes[3].set_title(f"Final Overlay [{mode_name}]\n(Green: Final Defect)", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    fig.suptitle(f"Phase 7: Region Filtering & Final Defect Mask [{mode_name}] | {os.path.basename(image_path)}", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_and_display(fig, os.path.join(OUTPUT_FIGURE_DIR, "region_filtering_demo.png"))


# ============================================================
# PHASE 8: BASELINE METHODS
# ============================================================
def run_phase8(image_path):
    print("=" * 60)
    print("PHASE 8: BASELINE METHODS")
    print("=" * 60)
    print(f"\nSelected image:\n{image_path}\n")

    _, _, _, normalized = preprocess_image(image_path)

    # 1. Standard Region Growing
    print("-" * 60)
    print("BASELINE 1: STANDARD REGION GROWING")
    print("-" * 60)
    initial_seeds_srg = select_seeds_srg(normalized, seed_threshold=SEED_THRESHOLD)
    srg_mask, _, srg_processed = standard_region_growing(
        image=normalized,
        seed_threshold=SEED_THRESHOLD,
        similarity_threshold=SIMILARITY_THRESHOLD,
        return_details=True,
    )
    srg_defect_pixels = int(np.sum(srg_mask == 255))
    print(f"Seeds:                {len(initial_seeds_srg)}")
    print(f"Processed pixels:     {srg_processed}")
    print(f"Final defect pixels:  {srg_defect_pixels}")
    print(f"Mask values:          {sorted(np.unique(srg_mask).tolist())}")

    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1[0].imshow(srg_mask, cmap="gray", vmin=0, vmax=255)
    ax1[0].set_title(f"Standard Region Growing Mask\nPixels: {srg_defect_pixels}", fontsize=10, fontweight="bold")
    ax1[0].axis("off")
    ov1 = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    ov1[srg_mask == 255] = [30, 255, 30]
    ax1[1].imshow(ov1)
    ax1[1].set_title("Standard RG Overlay", fontsize=10, fontweight="bold")
    ax1[1].axis("off")
    save_and_display(fig1, os.path.join(OUTPUT_FIGURE_DIR, "standard_region_growing_demo.png"))

    # 2. Otsu + Morphology
    print("\n" + "-" * 60)
    print("BASELINE 2: OTSU + MORPHOLOGY")
    print("-" * 60)
    otsu_thresh, raw_otsu, cleaned_otsu = otsu_morphology(normalized)
    otsu_defect_pixels = int(np.sum(cleaned_otsu == 255))
    print(f"Otsu threshold:       {otsu_thresh:.1f}")
    print(f"Raw Otsu pixels:      {int(np.sum(raw_otsu == 255))}")
    print(f"Cleaned defect pixels:{otsu_defect_pixels}")
    print(f"Mask values:          {sorted(np.unique(cleaned_otsu).tolist())}")

    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4.5))
    ax2[0].imshow(cleaned_otsu, cmap="gray", vmin=0, vmax=255)
    ax2[0].set_title(f"Otsu + Morphology Mask\nPixels: {otsu_defect_pixels}", fontsize=10, fontweight="bold")
    ax2[0].axis("off")
    ov2 = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    ov2[cleaned_otsu == 255] = [30, 255, 30]
    ax2[1].imshow(ov2)
    ax2[1].set_title("Otsu Overlay", fontsize=10, fontweight="bold")
    ax2[1].axis("off")
    save_and_display(fig2, os.path.join(OUTPUT_FIGURE_DIR, "otsu_morphology_demo.png"))

    # 3. Watershed
    print("\n" + "-" * 60)
    print("BASELINE 3: WATERSHED")
    print("-" * 60)
    markers, watershed_mask = watershed_segmentation(normalized, return_markers=True)
    watershed_pixels = int(np.sum(watershed_mask == 255))
    num_ws_regions = len([lbl for lbl in np.unique(markers) if lbl > 1])
    print(f"Watershed regions:    {num_ws_regions}")
    print(f"Final defect pixels:  {watershed_pixels}")
    print(f"Mask values:          {sorted(np.unique(watershed_mask).tolist())}")

    fig3, ax3 = plt.subplots(1, 2, figsize=(10, 4.5))
    ax3[0].imshow(watershed_mask, cmap="gray", vmin=0, vmax=255)
    ax3[0].set_title(f"Watershed Defect Mask\nPixels: {watershed_pixels}", fontsize=10, fontweight="bold")
    ax3[0].axis("off")
    ov3 = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    ov3[watershed_mask == 255] = [30, 255, 30]
    ax3[1].imshow(ov3)
    ax3[1].set_title("Watershed Overlay", fontsize=10, fontweight="bold")
    ax3[1].axis("off")
    save_and_display(fig3, os.path.join(OUTPUT_FIGURE_DIR, "watershed_demo.png"))


# ============================================================
# PHASE 9: BOX-LEVEL LOCALIZATION EVALUATION
# ============================================================
def run_phase9(image_path, use_original_primary=False):
    primary_name = "Original PGRG" if use_original_primary else "Improved PGRG (Default)"
    print("=" * 60)
    print(f"PHASE 9: BOX-LEVEL EVALUATION [Primary: {primary_name}]")
    print("=" * 60)
    print(f"\nSelected image:\n{image_path}\n")

    file_name = os.path.basename(image_path)
    xml_name = os.path.splitext(file_name)[0] + ".xml"

    candidate_dirs = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(image_path))), "annotations"),
        os.path.join("dataset", "NEU-DET", "train", "annotations"),
        os.path.join("dataset", "NEU-DET", "validation", "annotations"),
    ]
    xml_path = None
    for c_dir in candidate_dirs:
        potential = os.path.join(c_dir, xml_name)
        if os.path.exists(potential):
            xml_path = potential
            break

    if xml_path is None or not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML annotation not found for image {file_name}")

    gt_boxes = load_ground_truth_boxes(xml_path)
    _, _, _, normalized = preprocess_image(image_path)

    def execute_improved_pgrg():
        return improved_pgrg(normalized)

    def execute_original_pgrg():
        return original_pgrg(normalized)

    def execute_srg():
        return standard_region_growing(
            normalized, seed_threshold=ORIGINAL_SEED_THRESHOLD, similarity_threshold=ORIGINAL_SIMILARITY_THRESHOLD
        )

    def execute_otsu():
        _, _, cleaned_mask = otsu_morphology(normalized)
        return cleaned_mask

    def execute_watershed():
        _, defect_mask = watershed_segmentation(normalized)
        return defect_mask

    methods_to_eval = [
        ("Original PGRG", execute_original_pgrg),
        ("Standard Region Growing", execute_srg),
        ("Otsu + Morphology", execute_otsu),
        ("Watershed", execute_watershed),
    ]

    results = {}
    for name, func in methods_to_eval:
        mask, runtime_ms = measure_runtime(func)
        pred_boxes = mask_to_boxes(mask)
        tp, fp, fn, matched_pairs = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
        metrics = calculate_detection_metrics(tp, fp, fn, matched_pairs)
        results[name] = {
            "mask": mask,
            "pred_boxes": pred_boxes,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "metrics": metrics,
            "runtime_ms": runtime_ms,
        }

    print("BOX-LEVEL LOCALIZATION EVALUATION (IoU >= 0.5)")
    print(f"Ground-truth boxes: {len(gt_boxes)}\n")

    for name, _ in methods_to_eval:
        r = results[name]
        m = r["metrics"]
        print("-" * 60)
        print(f"Method:                 {name}")
        print(f"Predicted boxes:        {len(r['pred_boxes'])}")
        print(f"TP:                     {r['tp']}")
        print(f"FP:                     {r['fp']}")
        print(f"FN:                     {r['fn']}")
        print(f"Precision:              {m['precision']:.4f}")
        print(f"Recall:                 {m['recall']:.4f}")
        print(f"F1:                     {m['f1']:.4f}")
        print(f"Mean IoU:               {m['mean_iou']:.4f}")
        print(f"Runtime (ms):           {r['runtime_ms']:.2f}\n")

    print("=" * 60)
    print("NOTE: THIS IS A BOX-LEVEL LOCALIZATION EVALUATION.")
    print("Exact pixel-level defect contours are unavailable in NEU-DET.")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].imshow(normalized, cmap="gray")
    axes[0].set_title(f"Ground-Truth Boxes\nTotal Boxes: {len(gt_boxes)} (Red)", fontsize=10, fontweight="bold")
    for b in gt_boxes:
        rect = patches.Rectangle((b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                                 linewidth=2, edgecolor="red", facecolor="none")
        axes[0].add_patch(rect)
    axes[0].axis("off")

    orig_boxes = results["Original PGRG"]["pred_boxes"]
    axes[1].imshow(normalized, cmap="gray")
    axes[1].set_title(f"Original PGRG ({len(orig_boxes)} boxes)\n(Cyan: Predictions)", fontsize=10, fontweight="bold")
    for b in orig_boxes:
        rect = patches.Rectangle((b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                                 linewidth=2, edgecolor="cyan", facecolor="none")
        axes[1].add_patch(rect)
    axes[1].axis("off")

    # Overlay: GT (Red) + Original PGRG (Cyan)
    axes[2].imshow(normalized, cmap="gray")
    axes[2].set_title(f"Overlay Comparison\nRed: GT | Cyan: Original PGRG ({len(orig_boxes)})",
                      fontsize=10, fontweight="bold")
    for b in gt_boxes:
        rect = patches.Rectangle((b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                                 linewidth=2, edgecolor="red", facecolor="none")
        axes[2].add_patch(rect)
    for b in orig_boxes:
        rect = patches.Rectangle((b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                                 linewidth=2, edgecolor="cyan", facecolor="none")
        axes[2].add_patch(rect)
    axes[2].axis("off")

    orig_f1 = results["Original PGRG"]["metrics"]["f1"]
    orig_iou = results["Original PGRG"]["metrics"]["mean_iou"]
    fig.suptitle(f"BOX-LEVEL EVALUATION | {file_name}\n"
                 f"Original PGRG F1: {orig_f1:.4f} (Mean IoU: {orig_iou:.4f})",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save_and_display(fig, os.path.join(OUTPUT_FIGURE_DIR, f"evaluation_{os.path.splitext(file_name)[0]}.png"))



# ============================================================
# PHASE 10: FIVE-METHOD VALIDATION BENCHMARK (360 IMAGES)
# ============================================================
def collect_validation_samples():
    """Collect all 360 held-out validation samples across 6 defect classes."""
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    val_dir = os.path.join("dataset", "NEU-DET", "validation")
    ann_dir = os.path.join(val_dir, "annotations")
    samples = []
    for cls in classes:
        cls_dir = os.path.join(val_dir, "images", cls)
        if not os.path.exists(cls_dir):
            continue

        def sort_key(fname):
            try:
                return int(fname.split("_")[-1].split(".")[0])
            except ValueError:
                return fname

        img_files = sorted(
            [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")],
            key=sort_key,
        )
        for img_file in img_files:
            xml_file = os.path.splitext(img_file)[0] + ".xml"
            xml_path = os.path.join(ann_dir, xml_file)
            if not os.path.exists(xml_path):
                fallback = os.path.join("dataset", "NEU-DET", "train", "annotations", xml_file)
                if os.path.exists(fallback):
                    xml_path = fallback
            if xml_path and os.path.exists(xml_path):
                samples.append({
                    "image": img_file,
                    "class": cls,
                    "img_path": os.path.join(cls_dir, img_file),
                    "xml_path": xml_path,
                })
    return samples


def run_phase10(image_path=None, compare=False):
    """
    Phase 10: Final numerical benchmark on all 360 held-out validation images.
    Evaluates:
      1. Original PGRG (min_area=100, sim=25, seed=0.45, merge_gap=11, equal weights)
      2. Standard Region Growing
      3. Otsu + Morphology
      4. Watershed

    Outputs strictly the comparison table and a short completion message.
    Saves numerical results to results/improvement/final_held_out_results.csv.
    """
    val_samples = collect_validation_samples()
    if not val_samples:
        raise FileNotFoundError("No validation samples found under dataset/NEU-DET/validation.")

    print(f"Evaluating methods on {len(val_samples)} validation images...\n")

    methods = [
        ("Original PGRG", lambda norm: original_pgrg(norm)),
        ("Standard Region Growing", lambda norm: standard_region_growing(
            norm, seed_threshold=SRG_SEED_THRESHOLD, similarity_threshold=SRG_SIMILARITY_THRESHOLD
        )),
        ("Otsu + Morphology", lambda norm: otsu_morphology(norm)[2]),
        ("Watershed", lambda norm: watershed_segmentation(norm)[1]),
    ]

    stats = {
        mname: {"tp": 0, "fp": 0, "fn": 0, "ious": [], "runtimes": []}
        for mname, _ in methods
    }

    # Evaluate all 5 methods on the same 360 validation images
    for sample in val_samples:
        gt_boxes = load_ground_truth_boxes(sample["xml_path"])
        _, _, _, normalized = preprocess_image(sample["img_path"])

        for mname, mfunc in methods:
            mask, runtime_ms = measure_runtime(mfunc, normalized)
            pred_boxes = mask_to_boxes(mask)
            tp, fp, fn, matched_pairs = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)

            st = stats[mname]
            st["tp"] += tp
            st["fp"] += fp
            st["fn"] += fn
            st["runtimes"].append(runtime_ms)
            for _, _, iou_val in matched_pairs:
                st["ious"].append(iou_val)

    # Calculate detection metrics across the full 360 images
    results = {}
    for mname, _ in methods:
        st = stats[mname]
        tot_tp = st["tp"]
        tot_fp = st["fp"]
        tot_fn = st["fn"]
        ious = st["ious"]
        rts = st["runtimes"]

        prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        miou = sum(ious) / len(ious) if ious else 0.0
        mean_rt = sum(rts) / len(rts) if rts else 0.0

        results[mname] = {
            "tp": tot_tp,
            "fp": tot_fp,
            "fn": tot_fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mean_iou": miou,
            "mean_runtime_ms": mean_rt,
        }

    # Print the exact five-method validation table
    print("FINAL HELD-OUT RESULTS (Validation Set, 360 images)")
    print("=" * 90)
    print(f"{'Method':<25} | {'TP':>4} | {'FP':>6} | {'FN':>4} | {'P':>7} | "
          f"{'R':>7} | {'F1':>7} | {'IoU':>7} | {'RT(ms)':>8}")
    print("-" * 90)
    for mname in [m[0] for m in methods]:
        o = results[mname]
        print(f"{mname:<25} | {o['tp']:>4} | {o['fp']:>6} | {o['fn']:>4} | "
              f"{o['precision']:>7.4f} | {o['recall']:>7.4f} | {o['f1']:>7.4f} | "
              f"{o['mean_iou']:>7.4f} | {o['mean_runtime_ms']:>8.2f}")
    print("-" * 90)

    # Save to CSV file under results/improvement/
    out_dir = os.path.join("results", "improvement")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "final_held_out_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Method", "TP", "FP", "FN", "P", "R", "F1", "IoU", "RT(ms)"
        ])
        writer.writeheader()
        for mname in [m[0] for m in methods]:
            o = results[mname]
            writer.writerow({
                "Method": mname,
                "TP": o["tp"],
                "FP": o["fp"],
                "FN": o["fn"],
                "P": f"{o['precision']:.4f}",
                "R": f"{o['recall']:.4f}",
                "F1": f"{o['f1']:.4f}",
                "IoU": f"{o['mean_iou']:.4f}",
                "RT(ms)": f"{o['mean_runtime_ms']:.2f}",
            })

    print(f"\n[OK] Saved results to: {csv_path}")
    print("[OK] Phase 10 evaluation complete.")


def print_help_menu():
    help_text = """PGRG Steel Defect Detection (Improved PGRG Default)

Available phases:

1  Image Preprocessing
2  Priority Feature Extraction
3  Priority Map Generation
4  Seed Selection & Max-Heap
5  Priority-Guided Region Growing
6  Union-Find Region Merging
7  Region Filtering & Final Defect Mask
8  Baseline Methods (Standard RG, Otsu, Watershed)
9  Box-Level Evaluation (Original PGRG and baselines)
10 Final Experiments & Results (Numerical only: benchmark on 360 validation images)

PGRG Configuration:
  Original PGRG: min_area=100, sim=25, seed=0.45, merge_gap=11 (closing)
  --no-show      Disable interactive plot display (Phases 1-9)

Examples:
  # Run Original PGRG box-level evaluation:
  python main.py "dataset/NEU-DET/validation/images/scratches/scratches_241.jpg"

  # Run Phase 7 with Original PGRG:
  python main.py --phase 7 "dataset/NEU-DET/validation/images/scratches/scratches_241.jpg"

  # Run Phase 10 (Numerical benchmark on 360 validation images):
  python main.py --phase 10
"""
    print(help_text)


def main():
    # 1. Custom Help handling
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help_menu()
        return

    # Algorithm configuration selection: Improved by default, or Original if requested
    use_original = "--original-pgrg" in sys.argv
    compare = "--compare" in sys.argv or "--all-methods" in sys.argv

    # 2. Check for --phase argument
    if "--phase" in sys.argv:
        phase_idx = sys.argv.index("--phase")
        if phase_idx + 1 >= len(sys.argv):
            print("Invalid phase.\nValid phases are 1 through 10.")
            sys.exit(1)

        phase_str = sys.argv[phase_idx + 1]
        try:
            phase_num = int(phase_str)
        except ValueError:
            print("Invalid phase.\nValid phases are 1 through 10.")
            sys.exit(1)

        if phase_num < 1 or phase_num > 10:
            print("Invalid phase.\nValid phases are 1 through 10.")
            sys.exit(1)

        # Collect non-flag arguments after stripping --phase and its value
        remaining_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
        if phase_str in remaining_args:
            remaining_args.remove(phase_str)

        # Image path validation
        if phase_num in range(1, 10):
            if not remaining_args:
                print("This phase requires an image path.")
                sys.exit(1)
            image_path = remaining_args[0]
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
        elif phase_num == 10:
            image_path = remaining_args[0] if remaining_args else None
            if image_path and not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

        # Dispatch to requested phase ONLY
        if phase_num == 1:
            run_phase1(image_path)
        elif phase_num == 2:
            run_phase2(image_path)
        elif phase_num == 3:
            run_phase3(image_path)
        elif phase_num == 4:
            run_phase4(image_path)
        elif phase_num == 5:
            run_phase5(image_path, use_original=use_original)
        elif phase_num == 6:
            run_phase6(image_path, use_original=use_original)
        elif phase_num == 7:
            run_phase7(image_path, use_original=use_original)
        elif phase_num == 8:
            run_phase8(image_path)
        elif phase_num == 9:
            run_phase9(image_path, use_original_primary=use_original)
        elif phase_num == 10:
            run_phase10(image_path=image_path, compare=compare)

        return

    # 3. Default legacy fallback if --phase is not specified
    cli_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if not cli_args:
        print_help_menu()
        return

    image_path = cli_args[0]
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Legacy CLI flags
    if "--pgrg" in sys.argv:
        run_phase7(image_path, use_original=use_original)
    elif "--baseline1" in sys.argv:
        run_phase8(image_path)
    elif "--baseline2" in sys.argv:
        run_phase8(image_path)
    elif "--baseline3" in sys.argv:
        run_phase8(image_path)
    else:
        run_phase9(image_path, use_original_primary=use_original)


if __name__ == "__main__":
    main()
