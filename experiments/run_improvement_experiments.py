"""
run_improvement_experiments.py — Systematic PGRG Improvement Experiments

Phases:
  1. Parameter search on development subset (10 images/class = 60 images from train)
     - 1D sweeps of min_area, seed_threshold, similarity_threshold
  2. Region merging via morphological closing to reduce fragmentation
  3. Priority-map weight ablation
  4. Best-config selection and final evaluation on held-out validation set (360 images)

All results saved under results/improvement/.

Usage:
  python experiments/run_improvement_experiments.py --phase 1   # Parameter search
  python experiments/run_improvement_experiments.py --phase 2   # Region merging
  python experiments/run_improvement_experiments.py --phase 3   # Ablation
  python experiments/run_improvement_experiments.py --phase 4   # Final evaluation
  python experiments/run_improvement_experiments.py --phase all  # Run everything
"""

import os
import sys
import csv
import time
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_image
from src.priority_features import compute_priority_features
from src.priority_map import compute_priority_map
from src.pgrg import priority_guided_region_growing
from src.union_find import build_region_uf
from src.region_filtering import filter_regions
from src.region_growing import standard_region_growing
from src.otsu import otsu_morphology
from src.watershed import watershed_segmentation
from src.metrics import (
    load_ground_truth_boxes,
    mask_to_boxes,
    match_boxes,
    calculate_detection_metrics,
    measure_runtime,
)

# ── Configuration ─────────────────────────────────────────────────
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
DATASET_ROOT = os.path.join("dataset", "NEU-DET")
OUTPUT_DIR = os.path.join("results", "improvement")
IOU_THRESHOLD = 0.5
DEV_IMAGES_PER_CLASS = 10  # first 10 train images per class = 60 total

# Original baseline parameters
BASELINE_PARAMS = {
    "seed_threshold": 0.45,
    "similarity_threshold": 20,
    "min_area": 20,
}


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def sort_key(fname):
    """Deterministic numerical sort for image filenames."""
    try:
        return int(fname.split("_")[-1].split(".")[0])
    except ValueError:
        return fname


def collect_samples(split, max_per_class=None):
    """Collect image samples from a dataset split."""
    train_ann_dir = os.path.join(DATASET_ROOT, "train", "annotations")
    val_ann_dir = os.path.join(DATASET_ROOT, "validation", "annotations")

    samples = []
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_ROOT, split, "images", cls)
        if not os.path.exists(cls_dir):
            continue

        img_files = sorted(
            [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")],
            key=sort_key
        )
        if max_per_class is not None:
            img_files = img_files[:max_per_class]

        for img_file in img_files:
            img_path = os.path.join(cls_dir, img_file)
            xml_file = os.path.splitext(img_file)[0] + ".xml"
            xml_path = None
            for ann_dir in [train_ann_dir, val_ann_dir]:
                candidate = os.path.join(ann_dir, xml_file)
                if os.path.exists(candidate):
                    xml_path = candidate
                    break
            if xml_path:
                samples.append({
                    "image": img_file,
                    "class": cls,
                    "split": split,
                    "img_path": img_path,
                    "xml_path": xml_path,
                })
    return samples


def run_pgrg_pipeline(normalized, seed_threshold, similarity_threshold, min_area,
                      weights=None, merge_gap=0):
    """
    Run the full PGRG pipeline with configurable parameters.
    
    Args:
        normalized: preprocessed image
        seed_threshold: priority threshold for seed selection
        similarity_threshold: intensity difference threshold for region growing
        min_area: minimum connected-component area to keep
        weights: priority map feature weights (None = equal 1/3 each)
        merge_gap: morphological closing kernel size for merging nearby regions.
                   0 = no merging. Values like 3, 5, 7 close gaps of that size.
    
    Returns:
        binary defect mask (uint8, {0, 255})
    """
    _, idv, gm, tv = compute_priority_features(normalized)
    pmap = compute_priority_map(idv, gm, tv, weights=weights)
    rmask, _, _ = priority_guided_region_growing(
        normalized, pmap,
        seed_threshold=seed_threshold,
        similarity_threshold=similarity_threshold
    )
    uf, reg_idx = build_region_uf(rmask)
    filtered_mask, _ = filter_regions(rmask, uf, reg_idx, min_area=min_area)

    # Optional morphological closing to merge nearby fragments
    if merge_gap > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_gap, merge_gap))
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

    return filtered_mask


def evaluate_method(samples, method_func, method_name):
    """
    Evaluate a method on a set of samples.
    Returns per-image rows and aggregate stats per class.
    """
    rows = []
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0, "ious": [], "runtimes": []}
                   for c in CLASSES}

    for sample in samples:
        gt_boxes = load_ground_truth_boxes(sample["xml_path"])
        _, _, _, normalized = preprocess_image(sample["img_path"])

        mask, runtime_ms = measure_runtime(method_func, normalized)
        pred_boxes = mask_to_boxes(mask)
        tp, fp, fn, matched_pairs = match_boxes(pred_boxes, gt_boxes, iou_threshold=IOU_THRESHOLD)
        metrics = calculate_detection_metrics(tp, fp, fn, matched_pairs)

        rows.append({
            "image": sample["image"],
            "class": sample["class"],
            "split": sample["split"],
            "method": method_name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_pred_boxes": len(pred_boxes),
            "num_gt_boxes": len(gt_boxes),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "mean_iou": metrics["mean_iou"],
            "runtime_ms": runtime_ms,
        })

        st = class_stats[sample["class"]]
        st["tp"] += tp
        st["fp"] += fp
        st["fn"] += fn
        st["runtimes"].append(runtime_ms)
        for _, _, iou_val in matched_pairs:
            st["ious"].append(iou_val)

    # Compute aggregates
    tot_tp = sum(st["tp"] for st in class_stats.values())
    tot_fp = sum(st["fp"] for st in class_stats.values())
    tot_fn = sum(st["fn"] for st in class_stats.values())
    all_ious = []
    all_runtimes = []
    for st in class_stats.values():
        all_ious.extend(st["ious"])
        all_runtimes.extend(st["runtimes"])

    prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    miou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    mean_rt = sum(all_runtimes) / len(all_runtimes) if all_runtimes else 0.0

    overall = {
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "precision": prec, "recall": rec, "f1": f1,
        "mean_iou": miou, "mean_runtime_ms": mean_rt,
        "num_pred_boxes": sum(r["num_pred_boxes"] for r in rows),
    }

    # Per-class aggregates
    classwise = {}
    for c in CLASSES:
        st = class_stats[c]
        p = st["tp"] / (st["tp"] + st["fp"]) if (st["tp"] + st["fp"]) > 0 else 0.0
        r = st["tp"] / (st["tp"] + st["fn"]) if (st["tp"] + st["fn"]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        mi = sum(st["ious"]) / len(st["ious"]) if st["ious"] else 0.0
        mr = sum(st["runtimes"]) / len(st["runtimes"]) if st["runtimes"] else 0.0
        classwise[c] = {
            "tp": st["tp"], "fp": st["fp"], "fn": st["fn"],
            "precision": p, "recall": r, "f1": f,
            "mean_iou": mi, "mean_runtime_ms": mr,
        }

    return rows, overall, classwise


# ══════════════════════════════════════════════════════════════════
# PHASE 1: Parameter Search on Development Subset
# ══════════════════════════════════════════════════════════════════
def run_phase1():
    """1D parameter sweeps on development subset."""
    print("=" * 70)
    print("PHASE 1: PARAMETER SEARCH ON DEVELOPMENT SUBSET")
    print("=" * 70)

    dev_samples = collect_samples("train", max_per_class=DEV_IMAGES_PER_CLASS)
    print(f"Development subset: {len(dev_samples)} images "
          f"({DEV_IMAGES_PER_CLASS}/class × {len(CLASSES)} classes)")

    search_results = []

    # ── Sweep A: min_area (with baseline seed_threshold and similarity_threshold)
    print("\n--- Sweep A: min_area ---")
    min_area_values = [20, 30, 50, 75, 100]
    for ma in min_area_values:
        config_name = f"min_area={ma}"
        print(f"  Testing {config_name}...")

        def method(norm, _ma=ma):
            return run_pgrg_pipeline(norm,
                seed_threshold=0.45, similarity_threshold=20, min_area=_ma)

        _, overall, classwise = evaluate_method(dev_samples, method, config_name)
        search_results.append({
            "config": config_name,
            "sweep": "min_area",
            "seed_threshold": 0.45,
            "similarity_threshold": 20,
            "min_area": ma,
            "merge_gap": 0,
            "weights": "equal",
            **{k: overall[k] for k in ["tp", "fp", "fn", "precision", "recall", "f1",
                                         "mean_iou", "mean_runtime_ms", "num_pred_boxes"]},
        })
        print(f"    TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
              f"P={overall['precision']:.4f} R={overall['recall']:.4f} "
              f"F1={overall['f1']:.4f} IoU={overall['mean_iou']:.4f} "
              f"Boxes={overall['num_pred_boxes']} RT={overall['mean_runtime_ms']:.2f}ms")

    # ── Sweep B: seed_threshold (with baseline similarity and min_area)
    print("\n--- Sweep B: seed_threshold ---")
    seed_values = [0.45, 0.50, 0.55, 0.60, 0.65]
    for st in seed_values:
        config_name = f"seed_threshold={st}"
        print(f"  Testing {config_name}...")

        def method(norm, _st=st):
            return run_pgrg_pipeline(norm,
                seed_threshold=_st, similarity_threshold=20, min_area=20)

        _, overall, classwise = evaluate_method(dev_samples, method, config_name)
        search_results.append({
            "config": config_name,
            "sweep": "seed_threshold",
            "seed_threshold": st,
            "similarity_threshold": 20,
            "min_area": 20,
            "merge_gap": 0,
            "weights": "equal",
            **{k: overall[k] for k in ["tp", "fp", "fn", "precision", "recall", "f1",
                                         "mean_iou", "mean_runtime_ms", "num_pred_boxes"]},
        })
        print(f"    TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
              f"P={overall['precision']:.4f} R={overall['recall']:.4f} "
              f"F1={overall['f1']:.4f} IoU={overall['mean_iou']:.4f} "
              f"Boxes={overall['num_pred_boxes']} RT={overall['mean_runtime_ms']:.2f}ms")

    # ── Sweep C: similarity_threshold (with baseline seed and min_area)
    print("\n--- Sweep C: similarity_threshold ---")
    sim_values = [10, 15, 20, 25, 30]
    for sv in sim_values:
        config_name = f"similarity_threshold={sv}"
        print(f"  Testing {config_name}...")

        def method(norm, _sv=sv):
            return run_pgrg_pipeline(norm,
                seed_threshold=0.45, similarity_threshold=_sv, min_area=20)

        _, overall, classwise = evaluate_method(dev_samples, method, config_name)
        search_results.append({
            "config": config_name,
            "sweep": "similarity_threshold",
            "seed_threshold": 0.45,
            "similarity_threshold": sv,
            "min_area": 20,
            "merge_gap": 0,
            "weights": "equal",
            **{k: overall[k] for k in ["tp", "fp", "fn", "precision", "recall", "f1",
                                         "mean_iou", "mean_runtime_ms", "num_pred_boxes"]},
        })
        print(f"    TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
              f"P={overall['precision']:.4f} R={overall['recall']:.4f} "
              f"F1={overall['f1']:.4f} IoU={overall['mean_iou']:.4f} "
              f"Boxes={overall['num_pred_boxes']} RT={overall['mean_runtime_ms']:.2f}ms")

    # ── Sweep D: Combined promising configs based on 1D sweep insights
    # Pick the best min_area and best seed_threshold from sweeps A and B,
    # then test combinations
    print("\n--- Sweep D: Promising Combinations ---")

    # Identify best from each 1D sweep
    sweep_a = [r for r in search_results if r["sweep"] == "min_area"]
    sweep_b = [r for r in search_results if r["sweep"] == "seed_threshold"]
    sweep_c = [r for r in search_results if r["sweep"] == "similarity_threshold"]

    best_a = max(sweep_a, key=lambda r: r["f1"])
    best_b = max(sweep_b, key=lambda r: r["f1"])
    best_c = max(sweep_c, key=lambda r: r["f1"])

    print(f"  Best min_area from sweep A: {best_a['min_area']} (F1={best_a['f1']:.4f})")
    print(f"  Best seed_threshold from sweep B: {best_b['seed_threshold']} (F1={best_b['f1']:.4f})")
    print(f"  Best similarity from sweep C: {best_c['similarity_threshold']} (F1={best_c['f1']:.4f})")

    # Test top-2 from each sweep combined
    top2_ma = sorted(sweep_a, key=lambda r: r["f1"], reverse=True)[:2]
    top2_st = sorted(sweep_b, key=lambda r: r["f1"], reverse=True)[:2]
    top2_sv = sorted(sweep_c, key=lambda r: r["f1"], reverse=True)[:2]

    combos = set()
    for ma_r in top2_ma:
        for st_r in top2_st:
            for sv_r in top2_sv:
                combo = (st_r["seed_threshold"], sv_r["similarity_threshold"], ma_r["min_area"])
                # Skip the exact baseline
                if combo == (0.45, 20, 20):
                    continue
                combos.add(combo)

    for st_val, sv_val, ma_val in sorted(combos):
        config_name = f"st={st_val}_sim={sv_val}_ma={ma_val}"
        print(f"  Testing {config_name}...")

        def method(norm, _st=st_val, _sv=sv_val, _ma=ma_val):
            return run_pgrg_pipeline(norm,
                seed_threshold=_st, similarity_threshold=_sv, min_area=_ma)

        _, overall, _ = evaluate_method(dev_samples, method, config_name)
        search_results.append({
            "config": config_name,
            "sweep": "combination",
            "seed_threshold": st_val,
            "similarity_threshold": sv_val,
            "min_area": ma_val,
            "merge_gap": 0,
            "weights": "equal",
            **{k: overall[k] for k in ["tp", "fp", "fn", "precision", "recall", "f1",
                                         "mean_iou", "mean_runtime_ms", "num_pred_boxes"]},
        })
        print(f"    TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
              f"P={overall['precision']:.4f} R={overall['recall']:.4f} "
              f"F1={overall['f1']:.4f} IoU={overall['mean_iou']:.4f} "
              f"Boxes={overall['num_pred_boxes']} RT={overall['mean_runtime_ms']:.2f}ms")

    # Save parameter search CSV
    csv_path = os.path.join(OUTPUT_DIR, "parameter_search.csv")
    fieldnames = ["config", "sweep", "seed_threshold", "similarity_threshold", "min_area",
                  "merge_gap", "weights", "tp", "fp", "fn", "precision", "recall", "f1",
                  "mean_iou", "mean_runtime_ms", "num_pred_boxes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in search_results:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})

    print(f"\n[OK] Saved parameter search to: {csv_path}")
    print(f"Total configurations tested: {len(search_results)}")

    # Print ranking
    print("\n--- PARAMETER SEARCH RANKING (by F1) ---")
    ranked = sorted(search_results, key=lambda r: r["f1"], reverse=True)
    print(f"{'Rank':<5} {'Config':<35} {'F1':<8} {'P':<8} {'R':<8} {'IoU':<8} {'FP':<6} {'Boxes':<6}")
    for i, r in enumerate(ranked[:15], 1):
        print(f"{i:<5} {r['config']:<35} {r['f1']:<8.4f} {r['precision']:<8.4f} "
              f"{r['recall']:<8.4f} {r['mean_iou']:<8.4f} {r['fp']:<6} {r['num_pred_boxes']:<6}")

    return search_results


# ══════════════════════════════════════════════════════════════════
# PHASE 2: Region Merging Experiments
# ══════════════════════════════════════════════════════════════════
def run_phase2(best_params=None):
    """Test morphological closing to merge nearby fragmented regions."""
    print("\n" + "=" * 70)
    print("PHASE 2: REGION MERGING EXPERIMENTS")
    print("=" * 70)

    dev_samples = collect_samples("train", max_per_class=DEV_IMAGES_PER_CLASS)

    if best_params is None:
        # Read best params from phase 1 CSV if available
        csv_path = os.path.join(OUTPUT_DIR, "parameter_search.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            best_row = max(rows, key=lambda r: float(r["f1"]))
            best_params = {
                "seed_threshold": float(best_row["seed_threshold"]),
                "similarity_threshold": int(float(best_row["similarity_threshold"])),
                "min_area": int(float(best_row["min_area"])),
            }
            print(f"Using best params from Phase 1: {best_params}")
        else:
            best_params = BASELINE_PARAMS.copy()
            print(f"Using baseline params: {best_params}")

    merge_results = []

    # Test merge gap sizes with baseline params
    merge_gaps = [0, 3, 5, 7, 9, 11]
    for gap in merge_gaps:
        config_name = f"merge_gap={gap}"
        print(f"  Testing {config_name} with params {best_params}...")

        def method(norm, _gap=gap):
            return run_pgrg_pipeline(norm,
                seed_threshold=best_params["seed_threshold"],
                similarity_threshold=best_params["similarity_threshold"],
                min_area=best_params["min_area"],
                merge_gap=_gap)

        _, overall, _ = evaluate_method(dev_samples, method, config_name)
        merge_results.append({
            "config": config_name,
            "sweep": "merge_gap",
            "seed_threshold": best_params["seed_threshold"],
            "similarity_threshold": best_params["similarity_threshold"],
            "min_area": best_params["min_area"],
            "merge_gap": gap,
            "weights": "equal",
            **{k: overall[k] for k in ["tp", "fp", "fn", "precision", "recall", "f1",
                                         "mean_iou", "mean_runtime_ms", "num_pred_boxes"]},
        })
        print(f"    TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
              f"P={overall['precision']:.4f} R={overall['recall']:.4f} "
              f"F1={overall['f1']:.4f} IoU={overall['mean_iou']:.4f} "
              f"Boxes={overall['num_pred_boxes']} RT={overall['mean_runtime_ms']:.2f}ms")

    # Append to parameter search CSV
    csv_path = os.path.join(OUTPUT_DIR, "parameter_search.csv")
    fieldnames = ["config", "sweep", "seed_threshold", "similarity_threshold", "min_area",
                  "merge_gap", "weights", "tp", "fp", "fn", "precision", "recall", "f1",
                  "mean_iou", "mean_runtime_ms", "num_pred_boxes"]

    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        for row in merge_results:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})

    print(f"\n[OK] Appended merge results to: {csv_path}")

    # Visualize a few representative images
    visualize_merging(dev_samples, best_params, merge_gaps)

    return merge_results


def visualize_merging(samples, params, merge_gaps):
    """Visualize merging on representative images."""
    viz_images = ["scratches_1.jpg", "inclusion_1.jpg", "patches_1.jpg",
                  "pitted_surface_1.jpg", "crazing_1.jpg"]

    for target_name in viz_images:
        sample = None
        for s in samples:
            if s["image"] == target_name:
                sample = s
                break
        if sample is None:
            # Try to find in full train set
            for cls in CLASSES:
                img_path = os.path.join(DATASET_ROOT, "train", "images", cls, target_name)
                if os.path.exists(img_path):
                    xml_file = os.path.splitext(target_name)[0] + ".xml"
                    for ann_dir in [os.path.join(DATASET_ROOT, "train", "annotations"),
                                    os.path.join(DATASET_ROOT, "validation", "annotations")]:
                        xml_path = os.path.join(ann_dir, xml_file)
                        if os.path.exists(xml_path):
                            sample = {"image": target_name, "class": cls,
                                      "img_path": img_path, "xml_path": xml_path}
                            break
                if sample:
                    break
        if sample is None:
            continue

        _, _, _, normalized = preprocess_image(sample["img_path"])
        gt_boxes = load_ground_truth_boxes(sample["xml_path"])

        n_cols = min(len(merge_gaps) + 2, 6)  # original + GT + up to 4 merge configs
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

        # Column 0: Original image with GT boxes
        axes[0].imshow(normalized, cmap="gray")
        axes[0].set_title(f"GT Boxes ({len(gt_boxes)})", fontsize=9, fontweight="bold")
        import matplotlib.patches as mpatches
        for b in gt_boxes:
            rect = mpatches.Rectangle((b["xmin"], b["ymin"]),
                                      b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                                      linewidth=2, edgecolor="red", facecolor="none")
            axes[0].add_patch(rect)
        axes[0].axis("off")

        # Column 1: Original PGRG (no merge)
        mask_orig = run_pgrg_pipeline(normalized, **params, merge_gap=0)
        pred_orig = mask_to_boxes(mask_orig)
        tp_o, fp_o, fn_o, mp_o = match_boxes(pred_orig, gt_boxes, iou_threshold=0.5)
        m_o = calculate_detection_metrics(tp_o, fp_o, fn_o, mp_o)

        overlay = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        overlay[mask_orig == 255] = [30, 255, 30]
        axes[1].imshow(overlay)
        axes[1].set_title(f"No merge\nF1={m_o['f1']:.3f} B={len(pred_orig)}", fontsize=8, fontweight="bold")
        axes[1].axis("off")

        # Remaining columns: merge gaps
        for col_idx, gap in enumerate(merge_gaps[1:n_cols - 1], 2):
            mask_m = run_pgrg_pipeline(normalized, **params, merge_gap=gap)
            pred_m = mask_to_boxes(mask_m)
            tp_m, fp_m, fn_m, mp_m = match_boxes(pred_m, gt_boxes, iou_threshold=0.5)
            m_m = calculate_detection_metrics(tp_m, fp_m, fn_m, mp_m)

            overlay = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
            overlay[mask_m == 255] = [30, 255, 30]
            if col_idx < len(axes):
                axes[col_idx].imshow(overlay)
                axes[col_idx].set_title(f"Gap={gap}\nF1={m_m['f1']:.3f} B={len(pred_m)}",
                                       fontsize=8, fontweight="bold")
                axes[col_idx].axis("off")

        fig.suptitle(f"Region Merging: {target_name}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, f"merging_{os.path.splitext(target_name)[0]}.png")
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] Saved merging visualization: {fig_path}")


# ══════════════════════════════════════════════════════════════════
# PHASE 3: Priority-Map Weight Ablation
# ══════════════════════════════════════════════════════════════════
def run_phase3(best_params=None):
    """Test different priority map feature weight configurations."""
    print("\n" + "=" * 70)
    print("PHASE 3: PRIORITY MAP WEIGHT ABLATION")
    print("=" * 70)

    dev_samples = collect_samples("train", max_per_class=DEV_IMAGES_PER_CLASS)

    if best_params is None:
        csv_path = os.path.join(OUTPUT_DIR, "parameter_search.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                rows = list(csv.DictReader(f))
            best_row = max(rows, key=lambda r: float(r["f1"]))
            best_params = {
                "seed_threshold": float(best_row["seed_threshold"]),
                "similarity_threshold": int(float(best_row["similarity_threshold"])),
                "min_area": int(float(best_row["min_area"])),
            }
            print(f"Using best params: {best_params}")
        else:
            best_params = BASELINE_PARAMS.copy()

    # Weight configurations to test
    weight_configs = [
        ("equal (1/3, 1/3, 1/3)", None),
        ("deviation_only (1, 0, 0)",
         {"deviation": 1.0, "gradient": 0.0, "texture": 0.0}),
        ("gradient_only (0, 1, 0)",
         {"deviation": 0.0, "gradient": 1.0, "texture": 0.0}),
        ("texture_only (0, 0, 1)",
         {"deviation": 0.0, "gradient": 0.0, "texture": 1.0}),
        ("dev+grad (0.5, 0.5, 0)",
         {"deviation": 0.5, "gradient": 0.5, "texture": 0.0}),
        ("dev+tex (0.5, 0, 0.5)",
         {"deviation": 0.5, "gradient": 0.0, "texture": 0.5}),
        ("grad+tex (0, 0.5, 0.5)",
         {"deviation": 0.0, "gradient": 0.5, "texture": 0.5}),
        ("heavy_dev (0.5, 0.25, 0.25)",
         {"deviation": 0.5, "gradient": 0.25, "texture": 0.25}),
        ("heavy_grad (0.25, 0.5, 0.25)",
         {"deviation": 0.25, "gradient": 0.5, "texture": 0.25}),
        ("heavy_tex (0.25, 0.25, 0.5)",
         {"deviation": 0.25, "gradient": 0.25, "texture": 0.5}),
    ]

    ablation_results = []
    for wname, weights in weight_configs:
        config_name = f"weights={wname}"
        print(f"  Testing {config_name}...")

        def method(norm, _w=weights):
            return run_pgrg_pipeline(norm, weights=_w, **best_params, merge_gap=0)

        _, overall, classwise = evaluate_method(dev_samples, method, config_name)
        ablation_results.append({
            "config": config_name,
            "sweep": "weights",
            "seed_threshold": best_params["seed_threshold"],
            "similarity_threshold": best_params["similarity_threshold"],
            "min_area": best_params["min_area"],
            "merge_gap": 0,
            "weights": wname,
            **{k: overall[k] for k in ["tp", "fp", "fn", "precision", "recall", "f1",
                                         "mean_iou", "mean_runtime_ms", "num_pred_boxes"]},
            "classwise": classwise,
        })
        print(f"    TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
              f"P={overall['precision']:.4f} R={overall['recall']:.4f} "
              f"F1={overall['f1']:.4f} IoU={overall['mean_iou']:.4f}")

    # Append to parameter search CSV (without classwise sub-dict)
    csv_path = os.path.join(OUTPUT_DIR, "parameter_search.csv")
    fieldnames = ["config", "sweep", "seed_threshold", "similarity_threshold", "min_area",
                  "merge_gap", "weights", "tp", "fp", "fn", "precision", "recall", "f1",
                  "mean_iou", "mean_runtime_ms", "num_pred_boxes"]

    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        for row in ablation_results:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items() if k in fieldnames})

    print(f"\n[OK] Appended ablation results to: {csv_path}")

    # Print ranking
    print("\n--- ABLATION RANKING (by F1) ---")
    ranked = sorted(ablation_results, key=lambda r: r["f1"], reverse=True)
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['weights']:<30} F1={r['f1']:.4f} P={r['precision']:.4f} "
              f"R={r['recall']:.4f} IoU={r['mean_iou']:.4f}")

    return ablation_results


# ══════════════════════════════════════════════════════════════════
# PHASE 4: Final Evaluation on Held-Out Validation Set
# ══════════════════════════════════════════════════════════════════
def run_phase4():
    """Evaluate the best configuration and all baselines on the held-out validation set."""
    print("\n" + "=" * 70)
    print("PHASE 4: FINAL EVALUATION ON HELD-OUT VALIDATION SET (360 images)")
    print("=" * 70)

    val_samples = collect_samples("validation")
    print(f"Held-out validation set: {len(val_samples)} images")

    # Load best configuration from parameter search
    csv_path = os.path.join(OUTPUT_DIR, "parameter_search.csv")
    best_params = BASELINE_PARAMS.copy()
    best_merge_gap = 0
    best_weights = None
    best_weights_name = "equal"

    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            all_rows = list(csv.DictReader(f))
        if all_rows:
            best_row = max(all_rows, key=lambda r: float(r["f1"]))
            best_params = {
                "seed_threshold": float(best_row["seed_threshold"]),
                "similarity_threshold": int(float(best_row["similarity_threshold"])),
                "min_area": int(float(best_row["min_area"])),
            }
            best_merge_gap = int(float(best_row.get("merge_gap", 0)))
            best_weights_name = best_row.get("weights", "equal")

    # Parse weight name back to dict
    if best_weights_name == "equal" or "1/3" in best_weights_name:
        best_weights = None
    elif "deviation_only" in best_weights_name:
        best_weights = {"deviation": 1.0, "gradient": 0.0, "texture": 0.0}
    elif "gradient_only" in best_weights_name:
        best_weights = {"deviation": 0.0, "gradient": 1.0, "texture": 0.0}
    elif "texture_only" in best_weights_name:
        best_weights = {"deviation": 0.0, "gradient": 0.0, "texture": 1.0}
    elif "dev+grad" in best_weights_name:
        best_weights = {"deviation": 0.5, "gradient": 0.5, "texture": 0.0}
    elif "dev+tex" in best_weights_name:
        best_weights = {"deviation": 0.5, "gradient": 0.0, "texture": 0.5}
    elif "grad+tex" in best_weights_name:
        best_weights = {"deviation": 0.0, "gradient": 0.5, "texture": 0.5}
    elif "heavy_dev" in best_weights_name:
        best_weights = {"deviation": 0.5, "gradient": 0.25, "texture": 0.25}
    elif "heavy_grad" in best_weights_name:
        best_weights = {"deviation": 0.25, "gradient": 0.5, "texture": 0.25}
    elif "heavy_tex" in best_weights_name:
        best_weights = {"deviation": 0.25, "gradient": 0.25, "texture": 0.5}

    print(f"\nBest configuration from development tuning:")
    print(f"  seed_threshold = {best_params['seed_threshold']}")
    print(f"  similarity_threshold = {best_params['similarity_threshold']}")
    print(f"  min_area = {best_params['min_area']}")
    print(f"  merge_gap = {best_merge_gap}")
    print(f"  weights = {best_weights_name}")

    # Define all methods to evaluate
    def improved_pgrg(norm):
        return run_pgrg_pipeline(norm, **best_params,
                                weights=best_weights, merge_gap=best_merge_gap)

    def original_pgrg(norm):
        return run_pgrg_pipeline(norm, seed_threshold=0.45,
                                similarity_threshold=20, min_area=20)

    def srg(norm):
        return standard_region_growing(norm, seed_threshold=0.45, similarity_threshold=20)

    def otsu(norm):
        _, _, cleaned = otsu_morphology(norm)
        return cleaned

    def ws(norm):
        _, defect_mask = watershed_segmentation(norm)
        return defect_mask

    methods = [
        ("Improved PGRG", improved_pgrg),
        ("Original PGRG", original_pgrg),
        ("Standard Region Growing", srg),
        ("Otsu + Morphology", otsu),
        ("Watershed", ws),
    ]

    all_eval_rows = []
    all_overalls = {}
    all_classwise = {}

    for mname, mfunc in methods:
        print(f"\n  Evaluating: {mname}...")
        rows, overall, classwise = evaluate_method(val_samples, mfunc, mname)
        all_eval_rows.extend(rows)
        all_overalls[mname] = overall
        all_classwise[mname] = classwise
        print(f"    TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
              f"P={overall['precision']:.4f} R={overall['recall']:.4f} "
              f"F1={overall['f1']:.4f} IoU={overall['mean_iou']:.4f} "
              f"RT={overall['mean_runtime_ms']:.2f}ms")

    # ── Save evaluation_results.csv ──
    eval_csv = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    fieldnames = ["image", "class", "split", "method", "tp", "fp", "fn",
                  "num_pred_boxes", "num_gt_boxes",
                  "precision", "recall", "f1", "mean_iou", "runtime_ms"]
    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_eval_rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"\n[OK] Saved evaluation results to: {eval_csv}")

    # ── Save classwise_results.csv ──
    cw_csv = os.path.join(OUTPUT_DIR, "classwise_results.csv")
    cw_fields = ["class", "method", "tp", "fp", "fn", "precision", "recall", "f1",
                 "mean_iou", "mean_runtime_ms"]
    cw_rows = []
    for mname in [m[0] for m in methods]:
        for cls in CLASSES:
            cw = all_classwise[mname][cls]
            cw_rows.append({
                "class": cls,
                "method": mname,
                "tp": cw["tp"], "fp": cw["fp"], "fn": cw["fn"],
                "precision": f"{cw['precision']:.4f}",
                "recall": f"{cw['recall']:.4f}",
                "f1": f"{cw['f1']:.4f}",
                "mean_iou": f"{cw['mean_iou']:.4f}",
                "mean_runtime_ms": f"{cw['mean_runtime_ms']:.2f}",
            })
    with open(cw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cw_fields)
        writer.writeheader()
        writer.writerows(cw_rows)
    print(f"[OK] Saved classwise results to: {cw_csv}")

    # ── Generate comparison plots ──
    generate_comparison_plots(all_overalls, all_classwise)

    # ── Generate improvement report ──
    generate_improvement_report(best_params, best_merge_gap, best_weights_name,
                                all_overalls, all_classwise, val_samples)

    # ── Print final summary ──
    print("\n" + "=" * 70)
    print("FINAL HELD-OUT RESULTS (Validation Set, 360 images)")
    print("=" * 70)
    print(f"{'Method':<25} | {'TP':>4} | {'FP':>6} | {'FN':>4} | {'P':>7} | "
          f"{'R':>7} | {'F1':>7} | {'IoU':>7} | {'RT(ms)':>8}")
    print("-" * 90)
    for mname in [m[0] for m in methods]:
        o = all_overalls[mname]
        print(f"{mname:<25} | {o['tp']:>4} | {o['fp']:>6} | {o['fn']:>4} | "
              f"{o['precision']:>7.4f} | {o['recall']:>7.4f} | {o['f1']:>7.4f} | "
              f"{o['mean_iou']:>7.4f} | {o['mean_runtime_ms']:>8.2f}")


def generate_comparison_plots(overalls, classwise):
    """Generate comparison_plot.png and runtime_comparison.png."""
    method_names = list(overalls.keys())
    short_names = []
    for m in method_names:
        if "Improved" in m:
            short_names.append("Improved\nPGRG")
        elif "Original" in m:
            short_names.append("Original\nPGRG")
        elif "Standard" in m:
            short_names.append("Standard\nRG")
        elif "Otsu" in m:
            short_names.append("Otsu+\nMorph")
        else:
            short_names.append(m)

    colors = ["#2ecc71", "#3498db", "#ff7f0e", "#e74c3c", "#9b59b6"]

    # ── comparison_plot.png (4 subplots: P, R, F1, IoU) ──
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, metric, title in zip(axes,
                                  ["precision", "recall", "f1", "mean_iou"],
                                  ["Precision", "Recall", "F1 Score", "Mean Matched IoU"]):
        vals = [overalls[m][metric] for m in method_names]
        bars = ax.bar(short_names, vals, color=colors[:len(method_names)],
                      edgecolor="black", linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        y_max = max(vals) * 1.4 if max(vals) > 0 else 0.1
        ax.set_ylim(0, min(y_max, 1.0))
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + y_max * 0.02,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Held-Out Validation Results — Box-Level Localization (IoU ≥ 0.5)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "comparison_plot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved comparison plot")

    # ── runtime_comparison.png ──
    fig, ax = plt.subplots(figsize=(8, 5))
    runtimes = [overalls[m]["mean_runtime_ms"] for m in method_names]
    bars = ax.bar(short_names, runtimes, color=colors[:len(method_names)],
                  edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Mean Runtime (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Runtime Comparison — Per Image", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + max(runtimes) * 0.02,
                f"{h:.2f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "runtime_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved runtime comparison plot")

    # ── Class-wise F1 comparison ──
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CLASSES))
    width = 0.15
    for i, mname in enumerate(method_names):
        f1_vals = [classwise[mname][c]["f1"] for c in CLASSES]
        offset = (i - len(method_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1_vals, width, label=short_names[i].replace("\n", " "),
                      color=colors[i], edgecolor="black", linewidth=0.5)
        for b in bars:
            h = b.get_height()
            if h > 0.005:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax.set_title("Class-Wise F1 Comparison (Held-Out Validation)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "classwise_f1_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved classwise F1 comparison plot")


def generate_improvement_report(best_params, best_merge_gap, best_weights_name,
                                 overalls, classwise, val_samples):
    """Generate the improvement_report.md file."""
    report_path = os.path.join(OUTPUT_DIR, "improvement_report.md")

    orig = overalls.get("Original PGRG", {})
    impr = overalls.get("Improved PGRG", {})

    # Compute deltas
    def delta(key):
        o = orig.get(key, 0)
        i = impr.get(key, 0)
        return i - o

    lines = [
        "# PGRG Improvement Report",
        "",
        "## Experiment Setup",
        "",
        f"- **Development subset**: 60 images (10/class from train split)",
        f"- **Held-out evaluation**: {len(val_samples)} images (validation split)",
        f"- **Evaluation**: Box-level localization, IoU threshold = 0.5",
        f"- **Baseline parameters**: seed_threshold=0.45, similarity_threshold=20, min_area=20",
        "",
        "## Best Configuration Found",
        "",
        f"- **seed_threshold**: {best_params['seed_threshold']}",
        f"- **similarity_threshold**: {best_params['similarity_threshold']}",
        f"- **min_area**: {best_params['min_area']}",
        f"- **merge_gap**: {best_merge_gap}",
        f"- **weights**: {best_weights_name}",
        "",
        "## Held-Out Validation Results",
        "",
        "### Overall Comparison",
        "",
        "| Method | TP | FP | FN | Precision | Recall | F1 | Mean IoU | Runtime (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    method_order = ["Improved PGRG", "Original PGRG", "Standard Region Growing",
                    "Otsu + Morphology", "Watershed"]
    for m in method_order:
        if m in overalls:
            o = overalls[m]
            lines.append(f"| {m} | {o['tp']} | {o['fp']} | {o['fn']} | "
                         f"{o['precision']:.4f} | {o['recall']:.4f} | {o['f1']:.4f} | "
                         f"{o['mean_iou']:.4f} | {o['mean_runtime_ms']:.2f} |")

    lines.append("")

    if orig and impr:
        lines.extend([
            "### Improvement Deltas (Improved vs Original PGRG)",
            "",
            f"| Metric | Original | Improved | Delta |",
            f"|---|---:|---:|---:|",
            f"| Precision | {orig.get('precision',0):.4f} | {impr.get('precision',0):.4f} | {delta('precision'):+.4f} |",
            f"| Recall | {orig.get('recall',0):.4f} | {impr.get('recall',0):.4f} | {delta('recall'):+.4f} |",
            f"| F1 | {orig.get('f1',0):.4f} | {impr.get('f1',0):.4f} | {delta('f1'):+.4f} |",
            f"| Mean IoU | {orig.get('mean_iou',0):.4f} | {impr.get('mean_iou',0):.4f} | {delta('mean_iou'):+.4f} |",
            f"| FP Count | {orig.get('fp',0)} | {impr.get('fp',0)} | {impr.get('fp',0) - orig.get('fp',0):+d} |",
            f"| Runtime (ms) | {orig.get('mean_runtime_ms',0):.2f} | {impr.get('mean_runtime_ms',0):.2f} | {delta('mean_runtime_ms'):+.2f} |",
            "",
        ])

    lines.extend([
        "### Class-Wise F1 Scores",
        "",
        "| Class | " + " | ".join(method_order) + " |",
        "|---" + "|---:" * len(method_order) + "|",
    ])

    for cls in CLASSES:
        row = f"| {cls}"
        for m in method_order:
            if m in classwise:
                row += f" | {classwise[m][cls]['f1']:.4f}"
            else:
                row += " | —"
        row += " |"
        lines.append(row)

    lines.extend([
        "",
        "## Methodology",
        "",
        "1. **Phase 1**: 1D parameter sweeps (min_area, seed_threshold, similarity_threshold)",
        "   on 60-image development subset. Best values from each sweep combined.",
        "2. **Phase 2**: Morphological closing tested with gap sizes 0, 3, 5, 7, 9, 11.",
        "3. **Phase 3**: Priority-map weight ablation (10 configurations).",
        "4. **Phase 4**: Best configuration evaluated on held-out 360-image validation set.",
        "",
        "## Limitations",
        "",
        "- Development subset (60 images) is small; results may vary with different subsets.",
        "- Parameter search is 1D sweeps + top-2 combinations, not exhaustive grid search.",
        "- Morphological closing may merge unrelated nearby defects.",
        "- Evaluation is box-level localization, NOT pixel-level segmentation accuracy.",
        "- Original 1,800-image benchmark results are preserved unchanged.",
        "",
        "## Reproduction Commands",
        "",
        "```bash",
        "# Full pipeline (all phases):",
        "python experiments/run_improvement_experiments.py --phase all",
        "",
        "# Individual phases:",
        "python experiments/run_improvement_experiments.py --phase 1  # Parameter search",
        "python experiments/run_improvement_experiments.py --phase 2  # Region merging",
        "python experiments/run_improvement_experiments.py --phase 3  # Ablation",
        "python experiments/run_improvement_experiments.py --phase 4  # Final evaluation",
        "```",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Saved improvement report to: {report_path}")


# ══════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="PGRG Improvement Experiments")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3", "4", "all"],
                        help="Which phase to run (1, 2, 3, 4, or all)")
    parser.add_argument("--no-show", action="store_true", default=True,
                        help="Disable showing interactive GUI windows")
    args = parser.parse_args()

    ensure_dirs()

    if args.phase in ("1", "all"):
        search_results = run_phase1()

    if args.phase in ("2", "all"):
        run_phase2()

    if args.phase in ("3", "all"):
        run_phase3()

    if args.phase in ("4", "all"):
        run_phase4()

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
