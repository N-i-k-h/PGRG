"""
evaluate_1800_images.py — Full Dataset Benchmark (Part 3)

Evaluates all 1,800 images in the NEU-DET dataset across all 6 defect classes:
  1. crazing (300 images)
  2. inclusion (300 images)
  3. patches (300 images)
  4. pitted_surface (300 images)
  5. rolled-in_scale (300 images)
  6. scratches (300 images)

Methods benchmarked:
  - PGRG (Priority-Guided Region Growing)
  - Standard Region Growing (Baseline 1)
  - Otsu + Morphology (Baseline 2)
  - Watershed (Baseline 3)

Protocol:
  - Box-Level Localization Evaluation (IoU threshold = 0.5)

Outputs:
  - results/evaluation_1800_images.csv
  - results/evaluation_summary_1800_images.csv
  - results/figures/final_comparison_1800.png
  - results/figures/final_classwise_f1_1800.png
  - results/figures/final_classwise_iou_1800.png
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

SEED_THRESHOLD = 0.45
SIMILARITY_THRESHOLD = 20
MIN_AREA = 20

DATASET_ROOT = os.path.join("dataset", "NEU-DET")
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

DETAILED_CSV_PATH = os.path.join(RESULTS_DIR, "evaluation_1800_images.csv")
SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, "evaluation_summary_1800_images.csv")
FIG_COMPARISON_PATH = os.path.join(FIGURES_DIR, "final_comparison_1800.png")
FIG_CLASS_F1_PATH = os.path.join(FIGURES_DIR, "final_classwise_f1_1800.png")
FIG_CLASS_IOU_PATH = os.path.join(FIGURES_DIR, "final_classwise_iou_1800.png")


def collect_dataset_samples():
    """
    Collect all 1,800 images across train and validation splits,
    pairing each image with its XML annotation file.
    Searches both train/annotations and validation/annotations to handle
    the known dataset layout issue with crazing_240.xml.
    """
    train_ann_dir = os.path.join(DATASET_ROOT, "train", "annotations")
    val_ann_dir = os.path.join(DATASET_ROOT, "validation", "annotations")

    samples = []
    skipped = []

    # Iterate over splits and classes
    for split in ["train", "validation"]:
        split_img_dir = os.path.join(DATASET_ROOT, split, "images")
        for defect_cls in CLASSES:
            cls_img_dir = os.path.join(split_img_dir, defect_cls)
            if not os.path.exists(cls_img_dir):
                continue

            # Sort files deterministically (numerical index sorting)
            def sort_key(fname):
                try:
                    num_part = fname.split("_")[-1].split(".")[0]
                    return int(num_part)
                except ValueError:
                    return fname

            img_files = sorted(
                [f for f in os.listdir(cls_img_dir) if f.lower().endswith(".jpg")],
                key=sort_key
            )

            for img_file in img_files:
                img_path = os.path.join(cls_img_dir, img_file)
                xml_file = os.path.splitext(img_file)[0] + ".xml"

                # Check annotations in both train and validation directories
                xml_path = None
                train_xml_candidate = os.path.join(train_ann_dir, xml_file)
                val_xml_candidate = os.path.join(val_ann_dir, xml_file)

                if os.path.exists(train_xml_candidate):
                    xml_path = train_xml_candidate
                elif os.path.exists(val_xml_candidate):
                    xml_path = val_xml_candidate

                if xml_path is not None:
                    samples.append({
                        "image": img_file,
                        "class": defect_cls,
                        "split": split,
                        "img_path": img_path,
                        "xml_path": xml_path,
                    })
                else:
                    skipped.append(img_path)

    if skipped:
        print(f"[WARNING] Skipped {len(skipped)} images due to missing XML annotations:")
        for s in skipped[:5]:
            print(f"  - {s}")

    return samples


def run_full_benchmark():
    """Execute the full 1,800-image benchmark."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("EVALUATION STAGE — PART 3: FULL DATASET BENCHMARK (1,800 IMAGES)")
    print("BOX-LEVEL LOCALIZATION EVALUATION (IoU threshold = 0.5)")
    print("=" * 70)

    samples = collect_dataset_samples()
    num_samples = len(samples)
    print(f"Images collected for evaluation: {num_samples} (Expected: 1800)")

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

    # Process all images
    for idx, sample in enumerate(samples, 1):
        img_file = sample["image"]
        defect_cls = sample["class"]
        split = sample["split"]
        img_path = sample["img_path"]
        xml_path = sample["xml_path"]

        # 1. Load Ground Truth Boxes
        gt_boxes = load_ground_truth_boxes(xml_path)

        # 2. Preprocess Image
        _, _, _, normalized = preprocess_image(img_path)

        # 3. Evaluate each method
        for m_name, m_func in methods:
            mask, runtime_ms = measure_runtime(m_func, normalized)
            pred_boxes = mask_to_boxes(mask)
            tp, fp, fn, matched_pairs = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
            metrics = calculate_detection_metrics(tp, fp, fn, matched_pairs)

            detailed_rows.append({
                "image": img_file,
                "class": defect_cls,
                "split": split,
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

            # Aggregate stats
            st = method_class_stats[m_name][defect_cls]
            st["tp"] += tp
            st["fp"] += fp
            st["fn"] += fn
            st["count"] += 1
            st["runtimes"].append(runtime_ms)
            for _, _, iou_val in matched_pairs:
                st["ious"].append(iou_val)

        if idx % 100 == 0 or idx == num_samples:
            print(f"Progress: {idx:4d}/{num_samples} images evaluated ({idx*100//num_samples}%)...")

    # Save detailed CSV
    with open(DETAILED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image", "class", "split", "method", "tp", "fp", "fn",
            "precision", "recall", "f1", "mean_iou", "runtime_ms"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)

    print(f"\n[OK] Saved detailed results ({len(detailed_rows)} rows) to: {DETAILED_CSV_PATH}")

    # Compute Summary Statistics
    summary_rows = []
    overall_method_metrics = {}

    for m_name, _ in methods:
        tot_tp = 0
        tot_fp = 0
        tot_fn = 0
        all_ious = []
        all_runtimes = []

        # Class-wise rows
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
            "images": num_samples,
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

    # Generate Visualizations
    generate_visualizations(overall_method_metrics, summary_rows)

    # Print Final Console Report
    print_final_report(num_samples, overall_method_metrics, summary_rows)


def generate_visualizations(overall_metrics, summary_rows):
    """Generate the three required benchmark visualization figures."""
    method_keys = ["PGRG", "Standard Region Growing", "Otsu + Morphology", "Watershed"]
    short_names = ["PGRG", "Standard RG", "Otsu + Morph", "Watershed"]
    method_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # -------------------------------------------------------------
    # Figure 1: final_comparison_1800.png (5 subplots)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))

    precisions = [overall_metrics[m]["precision"] for m in method_keys]
    recalls = [overall_metrics[m]["recall"] for m in method_keys]
    f1s = [overall_metrics[m]["f1"] for m in method_keys]
    mious = [overall_metrics[m]["mean_iou"] for m in method_keys]
    runtimes = [overall_metrics[m]["mean_runtime_ms"] for m in method_keys]

    metric_data = [
        ("Precision", precisions, "#3498db", [0, max(precisions) * 1.35 if max(precisions) > 0 else 1.0]),
        ("Recall", recalls, "#e67e22", [0, max(recalls) * 1.35 if max(recalls) > 0 else 1.0]),
        ("F1 Score", f1s, "#2ecc71", [0, max(f1s) * 1.35 if max(f1s) > 0 else 1.0]),
        ("Mean IoU", mious, "#9b59b6", [0, 1.0]),
        ("Runtime (ms)", runtimes, "#e74c3c", [0, max(runtimes) * 1.25 if max(runtimes) > 0 else 10.0]),
    ]

    for ax, (title, vals, color, y_lim) in zip(axes, metric_data):
        bars = ax.bar(short_names, vals, color=color, width=0.55, edgecolor="black", linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(y_lim)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=25)
        for b in bars:
            h = b.get_height()
            label_text = f"{h:.4f}" if "Runtime" not in title else f"{h:.2f} ms"
            ax.text(b.get_x() + b.get_width() / 2.0, h + (y_lim[1] - y_lim[0]) * 0.02,
                    label_text, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.suptitle(
        "FULL DATASET BENCHMARK — 1,800 IMAGES (NEU-DET)\n"
        "BOX-LEVEL LOCALIZATION EVALUATION (IoU threshold = 0.5)",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig(FIG_COMPARISON_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved comparison visualization to: {FIG_COMPARISON_PATH}")

    # -------------------------------------------------------------
    # Figure 2: final_classwise_f1_1800.png (Class-wise F1)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(CLASSES))
    width = 0.20

    for i, (m_key, m_short, col) in enumerate(zip(method_keys, short_names, method_colors)):
        class_f1_vals = []
        for c in CLASSES:
            matched = [r for r in summary_rows if r["method"] == m_key and r["class"] == c]
            class_f1_vals.append(float(matched[0]["f1"]) if matched else 0.0)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, class_f1_vals, width, label=m_short, color=col, edgecolor="black", linewidth=0.6)
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2.0, h + 0.008, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7.5, rotation=0, fontweight="bold")

    ax.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax.set_title("Class-Wise Localization F1 Score Comparison (1,800 Images)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_CLASS_F1_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved class-wise F1 visualization to: {FIG_CLASS_F1_PATH}")

    # -------------------------------------------------------------
    # Figure 3: final_classwise_iou_1800.png (Class-wise Mean IoU)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, (m_key, m_short, col) in enumerate(zip(method_keys, short_names, method_colors)):
        class_iou_vals = []
        for c in CLASSES:
            matched = [r for r in summary_rows if r["method"] == m_key and r["class"] == c]
            class_iou_vals.append(float(matched[0]["mean_iou"]) if matched else 0.0)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, class_iou_vals, width, label=m_short, color=col, edgecolor="black", linewidth=0.6)
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2.0, h + 0.015, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7.5, rotation=0, fontweight="bold")

    ax.set_ylabel("Mean Matched IoU", fontsize=11, fontweight="bold")
    ax.set_title("Class-Wise Mean Matched-Box IoU Comparison (1,800 Images)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_CLASS_IOU_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved class-wise IoU visualization to: {FIG_CLASS_IOU_PATH}")


def print_final_report(num_samples, overall_metrics, summary_rows):
    """Print the required final console benchmark report."""
    method_keys = ["PGRG", "Standard Region Growing", "Otsu + Morphology", "Watershed"]
    short_names = ["PGRG", "Standard RG", "Otsu", "Watershed"]

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK REPORT — FULL NEU-DET DATASET")
    print("=" * 70)

    print(f"\nImages successfully evaluated:\n{num_samples}")
    print("\nExpected:\n1800\n")

    # 1. Overall Results Table
    print("-" * 70)
    print("OVERALL RESULTS")
    print("-" * 70)
    print(f"{'Method':<25s} | {'TP':>4s} | {'FP':>5s} | {'FN':>4s} | {'Precision':>9s} | {'Recall':>6s} | {'F1':>6s} | {'Mean IoU':>8s} | {'Runtime(ms)':>11s}")
    print("-" * 70)

    for m in method_keys:
        om = overall_metrics[m]
        print(
            f"{m:<25s} | "
            f"{om['tp']:>4d} | "
            f"{om['fp']:>5d} | "
            f"{om['fn']:>4d} | "
            f"{om['precision']:>9.4f} | "
            f"{om['recall']:>6.4f} | "
            f"{om['f1']:>6.4f} | "
            f"{om['mean_iou']:>8.4f} | "
            f"{om['mean_runtime_ms']:>11.2f}"
        )
    print("-" * 70)

    # 2. Class-Wise F1
    print("\n" + "-" * 70)
    print("CLASS-WISE F1")
    print("-" * 70)
    print(f"{'Class':<18s} | {'PGRG':>8s} | {'Standard RG':>11s} | {'Otsu':>8s} | {'Watershed':>9s}")
    print("-" * 70)

    for c in CLASSES:
        c_f1s = {}
        for m in method_keys:
            matched = [r for r in summary_rows if r["method"] == m and r["class"] == c]
            c_f1s[m] = matched[0]["f1"] if matched else "0.0000"
        print(
            f"{c:<18s} | "
            f"{c_f1s['PGRG']:>8s} | "
            f"{c_f1s['Standard Region Growing']:>11s} | "
            f"{c_f1s['Otsu + Morphology']:>8s} | "
            f"{c_f1s['Watershed']:>9s}"
        )
    print("-" * 70)

    # 3. Class-Wise Mean IoU
    print("\n" + "-" * 70)
    print("CLASS-WISE MEAN IoU")
    print("-" * 70)
    print(f"{'Class':<18s} | {'PGRG':>8s} | {'Standard RG':>11s} | {'Otsu':>8s} | {'Watershed':>9s}")
    print("-" * 70)

    for c in CLASSES:
        c_ious = {}
        for m in method_keys:
            matched = [r for r in summary_rows if r["method"] == m and r["class"] == c]
            c_ious[m] = matched[0]["mean_iou"] if matched else "0.0000"
        print(
            f"{c:<18s} | "
            f"{c_ious['PGRG']:>8s} | "
            f"{c_ious['Standard Region Growing']:>11s} | "
            f"{c_ious['Otsu + Morphology']:>8s} | "
            f"{c_ious['Watershed']:>9s}"
        )
    print("-" * 70)

    # 4. Runtime
    print("\n" + "-" * 70)
    print("RUNTIME")
    print("-" * 70)
    print(f"{'Method':<25s} | {'Mean Runtime(ms)':>16s}")
    print("-" * 70)
    for m in method_keys:
        print(f"{m:<25s} | {overall_metrics[m]['mean_runtime_ms']:>16.2f}")
    print("-" * 70)


if __name__ == "__main__":
    run_full_benchmark()
