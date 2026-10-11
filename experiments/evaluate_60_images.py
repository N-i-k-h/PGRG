"""
evaluate_60_images.py — Multi-Image Evaluation Experiment (Part 2)

Evaluates 10 images per defect class across all 6 NEU-DET classes (60 images total):
  1. crazing
  2. inclusion
  3. patches
  4. pitted_surface
  5. rolled-in_scale
  6. scratches

Methods benchmarked:
  - PGRG (Priority-Guided Region Growing)
  - Standard Region Growing (Baseline 1)
  - Otsu + Morphology (Baseline 2)
  - Watershed (Baseline 3)

Outputs:
  - results/evaluation_60_images.csv
  - results/evaluation_summary_60_images.csv
  - results/figures/evaluation_comparison_60_images.png
"""

import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import csv
import cv2
import numpy as np
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

# Configuration
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
IMAGES_PER_CLASS = 10
SEED_THRESHOLD = 0.45
SIMILARITY_THRESHOLD = 20
MIN_AREA = 20

TRAIN_IMG_DIR = os.path.join("dataset", "NEU-DET", "train", "images")
TRAIN_ANN_DIR = os.path.join("dataset", "NEU-DET", "train", "annotations")
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

DETAILED_CSV_PATH = os.path.join(RESULTS_DIR, "evaluation_60_images.csv")
SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, "evaluation_summary_60_images.csv")
COMPARISON_FIG_PATH = os.path.join(FIGURES_DIR, "evaluation_comparison_60_images.png")


def run_multi_image_experiment():
    """Execute the 60-image box-level localization evaluation experiment."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("EVALUATION STAGE — PART 2: MULTI-IMAGE EXPERIMENT (60 IMAGES)")
    print("BOX-LEVEL LOCALIZATION EVALUATION (IoU threshold = 0.5)")
    print("=" * 70)

    # 1. Collect deterministic list of 60 images (indices 1 to 10 for each class)
    selected_samples = []
    for defect_cls in CLASSES:
        for idx in range(1, IMAGES_PER_CLASS + 1):
            img_file = f"{defect_cls}_{idx}.jpg"
            xml_file = f"{defect_cls}_{idx}.xml"
            img_path = os.path.join(TRAIN_IMG_DIR, defect_cls, img_file)
            xml_path = os.path.join(TRAIN_ANN_DIR, xml_file)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image: {img_path}")
            if not os.path.exists(xml_path):
                raise FileNotFoundError(f"Missing annotation: {xml_path}")

            selected_samples.append({
                "class": defect_cls,
                "image_file": img_file,
                "img_path": img_path,
                "xml_path": xml_path,
            })

    print(f"Total samples to evaluate: {len(selected_samples)} across {len(CLASSES)} classes.\n")

    # Execution functions for each method
    def exec_pgrg(norm_img):
        lm, idv, gm, tv = compute_priority_features(norm_img)
        pmap = compute_priority_map(idv, gm, tv)
        rmask, _, _ = priority_guided_region_growing(
            norm_img, pmap, seed_threshold=SEED_THRESHOLD, similarity_threshold=SIMILARITY_THRESHOLD
        )
        uf, reg_idx = build_region_uf(rmask)
        filtered_mask, _ = filter_regions(rmask, uf, reg_idx, min_area=MIN_AREA)
        return filtered_mask

    def exec_srg(norm_img):
        return standard_region_growing(
            norm_img, seed_threshold=SEED_THRESHOLD, similarity_threshold=SIMILARITY_THRESHOLD
        )

    def exec_otsu(norm_img):
        _, _, cleaned_mask = otsu_morphology(norm_img)
        return cleaned_mask

    def exec_ws(norm_img):
        _, defect_mask = watershed_segmentation(norm_img)
        return defect_mask

    methods = [
        ("PGRG", exec_pgrg),
        ("Standard Region Growing", exec_srg),
        ("Otsu + Morphology", exec_otsu),
        ("Watershed", exec_ws),
    ]

    detailed_rows = []

    # Aggregation structures: method -> class -> stats
    method_class_stats = {
        m_name: {
            c_name: {
                "tp": 0, "fp": 0, "fn": 0, "ious": [], "runtimes": [], "count": 0
            }
            for c_name in CLASSES
        }
        for m_name, _ in methods
    }

    # Evaluate each sample
    for idx, sample in enumerate(selected_samples, 1):
        c_name = sample["class"]
        img_file = sample["image_file"]
        img_path = sample["img_path"]
        xml_path = sample["xml_path"]

        # Load GT boxes
        gt_boxes = load_ground_truth_boxes(xml_path)

        # Preprocess image
        _, _, _, normalized = preprocess_image(img_path)

        for m_name, m_func in methods:
            mask, runtime_ms = measure_runtime(m_func, normalized)
            pred_boxes = mask_to_boxes(mask)
            tp, fp, fn, matched_pairs = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
            metrics = calculate_detection_metrics(tp, fp, fn, matched_pairs)

            detailed_rows.append({
                "image": img_file,
                "class": c_name,
                "method": m_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": f"{metrics['precision']:.4f}",
                "recall": f"{metrics['recall']:.4f}",
                "f1": f"{metrics['f1']:.4f}",
                "mean_iou": f"{metrics['mean_iou']:.4f}",
                "runtime_ms": f"{runtime_ms:.2f}",
            })

            # Aggregate
            st = method_class_stats[m_name][c_name]
            st["tp"] += tp
            st["fp"] += fp
            st["fn"] += fn
            st["count"] += 1
            st["runtimes"].append(runtime_ms)
            for _, _, iou_val in matched_pairs:
                st["ious"].append(iou_val)

        if idx % 10 == 0:
            print(f"Processed {idx:2d}/60 images ({c_name} completed)...")

    # 2. Save detailed CSV
    with open(DETAILED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image", "class", "method", "tp", "fp", "fn",
            "precision", "recall", "f1", "mean_iou", "runtime_ms"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)

    print(f"\n[OK] Saved detailed results to: {DETAILED_CSV_PATH}")

    # 3. Compute Summary Table
    summary_rows = []
    overall_method_metrics = {}

    for m_name, _ in methods:
        tot_tp = 0
        tot_fp = 0
        tot_fn = 0
        all_ious = []
        all_runtimes = []

        for c_name in CLASSES:
            st = method_class_stats[m_name][c_name]
            tot_tp += st["tp"]
            tot_fp += st["fp"]
            tot_fn += st["fn"]
            all_ious.extend(st["ious"])
            all_runtimes.extend(st["runtimes"])

            prec = st["tp"] / (st["tp"] + st["fp"]) if (st["tp"] + st["fp"]) > 0 else 0.0
            rec = st["tp"] / (st["tp"] + st["fn"]) if (st["tp"] + st["fn"]) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            miou = sum(st["ious"]) / len(st["ious"]) if len(st["ious"]) > 0 else 0.0
            mean_rt = sum(st["runtimes"]) / len(st["runtimes"]) if len(st["runtimes"]) > 0 else 0.0

            summary_rows.append({
                "method": m_name,
                "class": c_name,
                "images": st["count"],
                "TP": st["tp"],
                "FP": st["fp"],
                "FN": st["fn"],
                "precision": f"{prec:.4f}",
                "recall": f"{rec:.4f}",
                "f1": f"{f1:.4f}",
                "mean_iou": f"{miou:.4f}",
                "mean_runtime_ms": f"{mean_rt:.2f}",
            })

        # Overall row for this method
        o_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        o_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        o_f1 = (2 * o_prec * o_rec / (o_prec + o_rec)) if (o_prec + o_rec) > 0 else 0.0
        o_miou = sum(all_ious) / len(all_ious) if len(all_ious) > 0 else 0.0
        o_mean_rt = sum(all_runtimes) / len(all_runtimes) if len(all_runtimes) > 0 else 0.0

        overall_method_metrics[m_name] = {
            "tp": tot_tp,
            "fp": tot_fp,
            "fn": tot_fn,
            "precision": o_prec,
            "recall": o_rec,
            "f1": o_f1,
            "mean_iou": o_miou,
            "mean_runtime_ms": o_mean_rt,
        }

        summary_rows.append({
            "method": m_name,
            "class": "OVERALL",
            "images": len(selected_samples),
            "TP": tot_tp,
            "FP": tot_fp,
            "FN": tot_fn,
            "precision": f"{o_prec:.4f}",
            "recall": f"{o_rec:.4f}",
            "f1": f"{o_f1:.4f}",
            "mean_iou": f"{o_miou:.4f}",
            "mean_runtime_ms": f"{o_mean_rt:.2f}",
        })

    # Save summary CSV
    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method", "class", "images", "TP", "FP", "FN",
            "precision", "recall", "f1", "mean_iou", "mean_runtime_ms"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[OK] Saved summary results to: {SUMMARY_CSV_PATH}")

    # 4. Create Comparison Visualization
    create_comparison_visualization(overall_method_metrics, COMPARISON_FIG_PATH)

    # 5. Print Clear Final Report
    print_final_report(overall_method_metrics, summary_rows)


def create_comparison_visualization(overall_metrics, output_path):
    """Generate high-quality visualization comparing all four methods."""
    method_names = ["PGRG", "Standard Region Growing", "Otsu + Morphology", "Watershed"]
    short_names = ["PGRG", "Standard RG", "Otsu + Morph", "Watershed"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    precisions = [overall_metrics[m]["precision"] for m in method_names]
    recalls = [overall_metrics[m]["recall"] for m in method_names]
    f1s = [overall_metrics[m]["f1"] for m in method_names]
    mious = [overall_metrics[m]["mean_iou"] for m in method_names]
    runtimes = [overall_metrics[m]["mean_runtime_ms"] for m in method_names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Subplot 1: Localization Quality Metrics (Precision, Recall, F1)
    x = np.arange(len(short_names))
    width = 0.25
    ax0 = axes[0]
    ax0.bar(x - width, precisions, width, label="Precision", color="#3498db")
    ax0.bar(x, recalls, width, label="Recall", color="#e67e22")
    ax0.bar(x + width, f1s, width, label="F1 Score", color="#2ecc71")
    ax0.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax0.set_title("Detection Metrics (IoU >= 0.5)", fontsize=12, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(short_names, fontsize=10, fontweight="bold")
    ax0.set_ylim(0, max(max(precisions), max(recalls), max(f1s), 0.5) * 1.25)
    ax0.legend(loc="upper left")
    ax0.grid(axis="y", linestyle="--", alpha=0.5)

    # Add labels on F1 bars
    for i, v in enumerate(f1s):
        ax0.text(i + width, v + 0.015, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    # Subplot 2: Mean Matched-Box IoU & Runtime
    ax1 = axes[1]
    bars_iou = ax1.bar(short_names, mious, color="#9b59b6", width=0.5)
    ax1.set_ylabel("Mean Matched IoU", fontsize=11, fontweight="bold")
    ax1.set_title("Mean Matched-Box IoU", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars_iou:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 0.02, f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Subplot 3: Mean Runtime (ms)
    ax2 = axes[2]
    bars_rt = ax2.bar(short_names, runtimes, color="#e74c3c", width=0.5)
    ax2.set_ylabel("Execution Time (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Mean Execution Time per Image", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars_rt:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h + max(runtimes) * 0.02, f"{h:.2f} ms", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(
        "MULTI-IMAGE EVALUATION BENCHMARK: 60 IMAGES (10 IMAGES / CLASS)\n"
        "BOX-LEVEL LOCALIZATION EVALUATION ON NEU-DET DATASET",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved comparison visualization to: {output_path}")


def print_final_report(overall_metrics, summary_rows):
    """Print the formatted final benchmark report."""
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK REPORT: MULTI-IMAGE EXPERIMENT (60 IMAGES)")
    print("=" * 70)

    # 1. Overall Summary Table
    print("\nOVERALL DETECTION RESULTS (60 IMAGES TOTAL):")
    print("-" * 75)
    headers = f"{'Method':<25s} | {'TP':>4s} | {'FP':>5s} | {'FN':>4s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s} | {'mIoU':>6s} | {'Time(ms)':>8s}"
    print(headers)
    print("-" * 75)

    for m_name in ["PGRG", "Standard Region Growing", "Otsu + Morphology", "Watershed"]:
        o = overall_metrics[m_name]
        row = (
            f"{m_name:<25s} | "
            f"{o['tp']:>4d} | "
            f"{o['fp']:>5d} | "
            f"{o['fn']:>4d} | "
            f"{o['precision']:>6.4f} | "
            f"{o['recall']:>6.4f} | "
            f"{o['f1']:>6.4f} | "
            f"{o['mean_iou']:>6.4f} | "
            f"{o['mean_runtime_ms']:>8.2f}"
        )
        print(row)
    print("-" * 75)

    # 2. Per-Class F1 Score Breakdown
    print("\nPER-CLASS F1 SCORE BREAKDOWN:")
    print("-" * 75)
    class_header = f"{'Defect Class':<18s} | {'PGRG':>8s} | {'Standard RG':>11s} | {'Otsu+Morph':>10s} | {'Watershed':>9s}"
    print(class_header)
    print("-" * 75)

    for c_name in CLASSES:
        c_f1s = {}
        for m_name in ["PGRG", "Standard Region Growing", "Otsu + Morphology", "Watershed"]:
            matched = [r for r in summary_rows if r["method"] == m_name and r["class"] == c_name]
            c_f1s[m_name] = matched[0]["f1"] if matched else "0.0000"

        c_row = (
            f"{c_name:<18s} | "
            f"{c_f1s['PGRG']:>8s} | "
            f"{c_f1s['Standard Region Growing']:>11s} | "
            f"{c_f1s['Otsu + Morphology']:>10s} | "
            f"{c_f1s['Watershed']:>9s}"
        )
        print(c_row)
    print("-" * 75)
    print("\n[NOTE] All metrics represent BOX-LEVEL LOCALIZATION EVALUATION (IoU threshold = 0.5).")
    print("=" * 70)


if __name__ == "__main__":
    run_multi_image_experiment()
