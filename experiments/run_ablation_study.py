"""
run_ablation_study.py — Ablation Study for PGRG Feature Contributions (Task A)

Evaluates how each priority-map feature contributes to PGRG performance by
comparing four configurations on a deterministic sample of 60 images
(10 per defect class):

  1. Intensity Deviation Only  (deviation=1.0, gradient=0.0, texture=0.0)
  2. Gradient Only              (deviation=0.0, gradient=1.0, texture=0.0)
  3. Texture Variance Only      (deviation=0.0, gradient=0.0, texture=1.0)
  4. Full PGRG                  (deviation=1/3, gradient=1/3, texture=1/3)

All other algorithm parameters are identical across configurations:
  - Seed threshold     = 0.45
  - Similarity threshold = 20
  - Min area            = 20
  - 8-neighbour connectivity
  - Box-level evaluation (IoU threshold = 0.5)

Runtime measurement includes priority-feature computation, priority-map
generation, region growing, union-find merging, and region filtering.
Preprocessing (load, grayscale, blur, normalize) is NOT included in runtime.

Outputs:
  results/ablation/ablation_results.csv      — per-image per-config results
  results/ablation/ablation_summary.csv      — aggregate per-config results
  results/ablation/ablation_comparison.png   — comparison bar charts
  results/ablation/ablation_report.md        — methodology & findings report
"""

import os
import sys
import csv
import time

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_image
from src.priority_features import compute_priority_features
from src.priority_map import compute_priority_map
from src.pgrg import priority_guided_region_growing
from src.union_find import build_region_uf
from src.region_filtering import filter_regions
from src.metrics import (
    load_ground_truth_boxes,
    mask_to_boxes,
    match_boxes,
    calculate_detection_metrics,
)

# ── Constants (matching existing benchmarks) ──────────────────────
SEED_THRESHOLD = 0.45
SIMILARITY_THRESHOLD = 20
MIN_AREA = 20
IOU_THRESHOLD = 0.5
IMAGES_PER_CLASS = 10

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Ablation configurations ─────────────────────────────────────────
# Each entry: (display_name, weight_dict)
# weight_dict is passed to compute_priority_map(weights=...)
ABLATION_CONFIGS = [
    (
        "Deviation Only",
        {"deviation": 1.0, "gradient": 0.0, "texture": 0.0},
    ),
    (
        "Gradient Only",
        {"deviation": 0.0, "gradient": 1.0, "texture": 0.0},
    ),
    (
        "Texture Only",
        {"deviation": 0.0, "gradient": 0.0, "texture": 1.0},
    ),
    (
        "Full PGRG",
        None,  # None → default equal weights (1/3, 1/3, 1/3)
    ),
]

DATASET_ROOT = os.path.join("dataset", "NEU-DET")
OUTPUT_DIR = os.path.join("results", "ablation")


# ── Sample collection ─────────────────────────────────────────────
def collect_ablation_samples():
    """
    Collect exactly IMAGES_PER_CLASS images per defect class.

    Selection is deterministic: images are sorted by their numeric index
    (e.g. crazing_1.jpg → 1) and the first IMAGES_PER_CLASS per class
    are selected. Both train and validation splits are searched for
    annotations.
    """
    train_img_root = os.path.join(DATASET_ROOT, "train", "images")
    val_img_root = os.path.join(DATASET_ROOT, "validation", "images")
    train_ann_dir = os.path.join(DATASET_ROOT, "train", "annotations")
    val_ann_dir = os.path.join(DATASET_ROOT, "validation", "annotations")

    samples = []
    skipped = []

    for defect_cls in CLASSES:
        cls_samples = []

        # Gather images from both splits
        for split, img_root in [("train", train_img_root), ("validation", val_img_root)]:
            cls_dir = os.path.join(img_root, defect_cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith(".jpg"):
                    continue
                img_path = os.path.join(cls_dir, fname)
                xml_name = os.path.splitext(fname)[0] + ".xml"

                # Look for annotation in both annotation directories
                xml_path = None
                for ann_dir in [train_ann_dir, val_ann_dir]:
                    candidate = os.path.join(ann_dir, xml_name)
                    if os.path.exists(candidate):
                        xml_path = candidate
                        break

                if xml_path is not None:
                    cls_samples.append({
                        "image": fname,
                        "class": defect_cls,
                        "split": split,
                        "img_path": img_path,
                        "xml_path": xml_path,
                    })
                else:
                    skipped.append(img_path)

        # Sort deterministically by numeric index in filename
        def sort_key(s):
            try:
                return int(s["image"].split("_")[-1].split(".")[0])
            except ValueError:
                return s["image"]

        cls_samples.sort(key=sort_key)

        # Take the first IMAGES_PER_CLASS
        selected = cls_samples[:IMAGES_PER_CLASS]
        if len(selected) < IMAGES_PER_CLASS:
            print(f"[WARNING] {defect_cls}: only {len(selected)}/{IMAGES_PER_CLASS} images available")

        samples.extend(selected)

    if skipped:
        print(f"[WARNING] Skipped {len(skipped)} images due to missing XML annotations")

    return samples


# ── PGRG execution with configurable weights ──────────────────────
def execute_pgrg_variant(normalized_image, weights):
    """
    Run the full PGRG pipeline on a single image with the specified
    priority-map weights.

    Args:
        normalized_image: Preprocessed image from preprocess_image().
        weights: Weight dict or None (for default 1/3 each).

    Returns:
        filtered_mask: Final binary defect mask (0/255).
    """
    # Feature extraction
    _, intensity_dev, grad_mag, texture_var = compute_priority_features(normalized_image)

    # Priority map with configurable weights
    priority_map = compute_priority_map(
        intensity_deviation=intensity_dev,
        gradient_magnitude=grad_mag,
        texture_variance=texture_var,
        weights=weights,
    )

    # Region growing
    region_mask, _, _ = priority_guided_region_growing(
        image=normalized_image,
        priority_map=priority_map,
        seed_threshold=SEED_THRESHOLD,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    # Union-Find merging
    uf, region_indices = build_region_uf(region_mask)

    # Region filtering
    filtered_mask, _ = filter_regions(
        region_mask=region_mask,
        uf=uf,
        region_indices=region_indices,
        min_area=MIN_AREA,
    )

    return filtered_mask


# ── Main benchmark ────────────────────────────────────────────────
def run_ablation_study():
    """Execute the ablation study on 60 deterministic images."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("ABLATION STUDY — PGRG Feature Contribution Analysis")
    print("Box-Level Localization Evaluation (IoU threshold = 0.5)")
    print("=" * 70)

    # 1. Collect samples
    samples = collect_ablation_samples()
    print(f"\nImages collected: {len(samples)} (expected: {IMAGES_PER_CLASS * len(CLASSES)})")
    for cls in CLASSES:
        count = sum(1 for s in samples if s["class"] == cls)
        print(f"  {cls}: {count} images")

    # Verify sample count
    if len(samples) != IMAGES_PER_CLASS * len(CLASSES):
        print(f"[ERROR] Expected {IMAGES_PER_CLASS * len(CLASSES)} images, got {len(samples)}")
        sys.exit(1)

    # 2. Verify all XML annotations exist
    print("\nVerifying annotations...")
    for sample in samples:
        if not os.path.exists(sample["xml_path"]):
            print(f"[ERROR] Missing annotation: {sample['xml_path']}")
            sys.exit(1)
        gt_boxes = load_ground_truth_boxes(sample["xml_path"])
        if len(gt_boxes) == 0:
            print(f"[WARNING] No ground-truth boxes in: {sample['xml_path']}")
    print("[OK] All 60 XML annotations verified.\n")

    # 3. Print configuration summary
    print("Ablation configurations:")
    for config_name, weights in ABLATION_CONFIGS:
        if weights is None:
            w_str = "deviation=1/3, gradient=1/3, texture=1/3 (default)"
        else:
            w_str = ", ".join(f"{k}={v}" for k, v in weights.items())
        print(f"  - {config_name}: {w_str}")

    print(f"\nAlgorithm parameters:")
    print(f"  Seed threshold       = {SEED_THRESHOLD}")
    print(f"  Similarity threshold = {SIMILARITY_THRESHOLD}")
    print(f"  Min area             = {MIN_AREA}")
    print(f"  IoU threshold        = {IOU_THRESHOLD}")

    # 4. Run evaluation
    detailed_rows = []

    # Aggregation: config_name → {tp, fp, fn, ious, runtimes}
    agg = {
        name: {"tp": 0, "fp": 0, "fn": 0, "ious": [], "runtimes": []}
        for name, _ in ABLATION_CONFIGS
    }

    total_evals = len(samples) * len(ABLATION_CONFIGS)
    eval_count = 0

    print(f"\nRunning {total_evals} evaluations ({len(samples)} images × {len(ABLATION_CONFIGS)} configs)...\n")

    for sample in samples:
        img_path = sample["img_path"]
        xml_path = sample["xml_path"]
        defect_cls = sample["class"]
        img_name = sample["image"]

        # Preprocess once (shared across all configurations)
        _, _, _, normalized = preprocess_image(img_path)

        # Load ground truth once
        gt_boxes = load_ground_truth_boxes(xml_path)

        for config_name, weights in ABLATION_CONFIGS:
            # Measure runtime (includes feature extraction + priority map + growing + filtering)
            t_start = time.perf_counter()
            mask = execute_pgrg_variant(normalized, weights)
            t_end = time.perf_counter()
            runtime_ms = (t_end - t_start) * 1000.0

            # Box-level evaluation
            pred_boxes = mask_to_boxes(mask)
            tp, fp, fn, matched_pairs = match_boxes(pred_boxes, gt_boxes, iou_threshold=IOU_THRESHOLD)
            metrics = calculate_detection_metrics(tp, fp, fn, matched_pairs)

            # Record detailed result
            detailed_rows.append({
                "image": img_name,
                "class": defect_cls,
                "config": config_name,
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
            agg[config_name]["tp"] += tp
            agg[config_name]["fp"] += fp
            agg[config_name]["fn"] += fn
            agg[config_name]["runtimes"].append(runtime_ms)
            for _, _, iou_val in matched_pairs:
                agg[config_name]["ious"].append(iou_val)

            eval_count += 1

        # Progress
        img_idx = samples.index(sample) + 1
        if img_idx % 10 == 0 or img_idx == len(samples):
            print(f"  Progress: {img_idx}/{len(samples)} images ({eval_count}/{total_evals} evaluations)")

    # 5. Save detailed CSV
    detailed_csv_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    fieldnames = [
        "image", "class", "config", "tp", "fp", "fn",
        "precision", "recall", "f1", "mean_iou", "runtime_ms",
    ]
    with open(detailed_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"\n[OK] Saved detailed results ({len(detailed_rows)} rows): {detailed_csv_path}")

    # 6. Compute and save summary
    summary_rows = []
    for config_name, _ in ABLATION_CONFIGS:
        st = agg[config_name]
        tp_total = st["tp"]
        fp_total = st["fp"]
        fn_total = st["fn"]

        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        miou = sum(st["ious"]) / len(st["ious"]) if len(st["ious"]) > 0 else 0.0
        mean_rt = sum(st["runtimes"]) / len(st["runtimes"]) if len(st["runtimes"]) > 0 else 0.0

        summary_rows.append({
            "config": config_name,
            "images": len(samples),
            "TP": tp_total,
            "FP": fp_total,
            "FN": fn_total,
            "precision": f"{prec:.4f}",
            "recall": f"{rec:.4f}",
            "f1": f"{f1:.4f}",
            "mean_iou": f"{miou:.4f}",
            "mean_runtime_ms": f"{mean_rt:.2f}",
        })

    summary_csv_path = os.path.join(OUTPUT_DIR, "ablation_summary.csv")
    summary_fields = [
        "config", "images", "TP", "FP", "FN",
        "precision", "recall", "f1", "mean_iou", "mean_runtime_ms",
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[OK] Saved summary results: {summary_csv_path}")

    # 7. Generate comparison figure
    generate_ablation_figure(summary_rows)

    # 8. Generate report
    generate_ablation_report(samples, summary_rows, detailed_rows)

    # 9. Print console summary
    print_console_summary(summary_rows)

    # 10. Verification checks
    run_verification_checks(samples, detailed_rows, summary_rows)

    chart_path = os.path.join(OUTPUT_DIR, "ablation_comparison.png")
    print("\nAblation study completed successfully:")
    print(f"  Chart path       : {chart_path}")
    print(f"  Results CSV path : {detailed_csv_path}")
    print(f"  Summary CSV path : {summary_csv_path}")


# ── Visualization ─────────────────────────────────────────────────
def generate_ablation_figure(summary_rows):
    """Create comparison bar charts for the ablation study."""
    config_names = [r["config"] for r in summary_rows]
    short_names = ["Dev Only", "Grad Only", "Tex Only", "Full PGRG"]

    precisions = [float(r["precision"]) for r in summary_rows]
    recalls = [float(r["recall"]) for r in summary_rows]
    f1s = [float(r["f1"]) for r in summary_rows]
    mious = [float(r["mean_iou"]) for r in summary_rows]
    runtimes = [float(r["mean_runtime_ms"]) for r in summary_rows]

    # Colors: single-feature variants in muted tones, Full PGRG highlighted
    colors = ["#5dade2", "#f39c12", "#58d68d", "#e74c3c"]

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5))

    metric_data = [
        ("Precision", precisions),
        ("Recall", recalls),
        ("F1 Score", f1s),
        ("Mean Matched IoU", mious),
        ("Runtime (ms)", runtimes),
    ]

    for ax, (title, vals) in zip(axes, metric_data):
        max_val = max(vals) if max(vals) > 0 else 1.0
        y_top = max_val * 1.4 if "Runtime" not in title else max_val * 1.3

        bars = ax.bar(
            short_names, vals,
            color=colors, width=0.55,
            edgecolor="black", linewidth=0.8,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, y_top)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=20, labelsize=9)

        for bar in bars:
            h = bar.get_height()
            fmt = f"{h:.4f}" if "Runtime" not in title else f"{h:.1f} ms"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + y_top * 0.02,
                fmt,
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
            )

    fig.suptitle(
        "ABLATION STUDY — PGRG Feature Contribution Analysis\n"
        "Box-Level Localization Evaluation (60 images, IoU ≥ 0.5)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_path = os.path.join(OUTPUT_DIR, "ablation_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    if os.path.exists(fig_path) and os.path.getsize(fig_path) > 0:
        print(f"[OK] Saved comparison figure ({os.path.getsize(fig_path)} bytes): {fig_path}")
    else:
        print(f"[ERROR] Failed to save comparison figure: {fig_path}")


# ── Report generation ─────────────────────────────────────────────
def generate_ablation_report(samples, summary_rows, detailed_rows):
    """Generate the ablation study report in Markdown."""
    report_path = os.path.join(OUTPUT_DIR, "ablation_report.md")

    # Determine best config per metric
    f1s = {r["config"]: float(r["f1"]) for r in summary_rows}
    precs = {r["config"]: float(r["precision"]) for r in summary_rows}
    recs = {r["config"]: float(r["recall"]) for r in summary_rows}
    ious = {r["config"]: float(r["mean_iou"]) for r in summary_rows}
    rts = {r["config"]: float(r["mean_runtime_ms"]) for r in summary_rows}

    best_f1 = max(f1s, key=f1s.get) if any(v > 0 for v in f1s.values()) else "N/A"
    best_prec = max(precs, key=precs.get) if any(v > 0 for v in precs.values()) else "N/A"
    best_rec = max(recs, key=recs.get) if any(v > 0 for v in recs.values()) else "N/A"
    best_iou = max(ious, key=ious.get) if any(v > 0 for v in ious.values()) else "N/A"

    # Build class distribution string
    class_dist = ", ".join(
        f"{cls}: {sum(1 for s in samples if s['class'] == cls)}"
        for cls in CLASSES
    )

    lines = []
    lines.append("# Ablation Study Report — PGRG Feature Contribution Analysis")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append("")
    lines.append("Measure how each individual priority-map feature contributes to the")
    lines.append("overall PGRG defect detection performance. This ablation study isolates")
    lines.append("three single-feature variants and compares them against the full")
    lines.append("three-feature combination used by the default PGRG configuration.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Configurations")
    lines.append("")
    lines.append("| # | Configuration | Deviation Weight | Gradient Weight | Texture Weight |")
    lines.append("|---|---|---|---|---|")
    lines.append("| 1 | Deviation Only | **1.0** | 0.0 | 0.0 |")
    lines.append("| 2 | Gradient Only | 0.0 | **1.0** | 0.0 |")
    lines.append("| 3 | Texture Only | 0.0 | 0.0 | **1.0** |")
    lines.append("| 4 | Full PGRG | 1/3 | 1/3 | 1/3 |")
    lines.append("")
    lines.append("The Full PGRG configuration uses `weights=None`, which defaults to equal")
    lines.append("weights (1/3 each) in `compute_priority_map()`. This is identical to the")
    lines.append("configuration used in all existing benchmark evaluations.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Methodology")
    lines.append("")
    lines.append("### Sample Selection")
    lines.append(f"- **Total images**: {len(samples)}")
    lines.append(f"- **Images per class**: {IMAGES_PER_CLASS}")
    lines.append(f"- **Class distribution**: {class_dist}")
    lines.append("- **Selection**: First 10 images per class, sorted by numeric index")
    lines.append("  (deterministic, reproducible)")
    lines.append("")
    lines.append("### Shared Parameters")
    lines.append(f"- Seed threshold: {SEED_THRESHOLD}")
    lines.append(f"- Similarity threshold: {SIMILARITY_THRESHOLD}")
    lines.append(f"- Minimum area: {MIN_AREA}")
    lines.append(f"- IoU threshold: {IOU_THRESHOLD}")
    lines.append("- Connectivity: 8-neighbour")
    lines.append("")
    lines.append("### Evaluation Protocol")
    lines.append("- **Type**: Box-level localization evaluation")
    lines.append("- Predicted defect masks are converted to bounding boxes via connected")
    lines.append("  component analysis (`mask_to_boxes`).")
    lines.append("- Greedy one-to-one matching between predicted and ground-truth boxes")
    lines.append("  with IoU ≥ 0.5.")
    lines.append("- Metrics: Precision, Recall, F1, Mean Matched IoU.")
    lines.append("- Ground truth: PASCAL VOC XML bounding-box annotations from NEU-DET.")
    lines.append("")
    lines.append("### Runtime Measurement")
    lines.append("- **Included in runtime**: Priority feature extraction, priority map")
    lines.append("  computation, region growing, union-find merging, region filtering.")
    lines.append("- **Excluded from runtime**: Image loading, grayscale conversion,")
    lines.append("  Gaussian blur, normalization (preprocessing).")
    lines.append("- Measured using `time.perf_counter()` for high precision.")
    lines.append("")
    lines.append("> **Note**: These are box-level localization metrics, not pixel-level")
    lines.append("> segmentation metrics. NEU-DET provides only bounding-box annotations.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Results")
    lines.append("")
    lines.append("### Overall Performance")
    lines.append("")
    lines.append("| Configuration | Images | TP | FP | FN | Precision | Recall | F1 | Mean IoU | Runtime (ms) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in summary_rows:
        lines.append(
            f"| {r['config']} | {r['images']} | {r['TP']} | {r['FP']} | {r['FN']} | "
            f"{r['precision']} | {r['recall']} | {r['f1']} | {r['mean_iou']} | {r['mean_runtime_ms']} |"
        )
    lines.append("")

    lines.append("### Best Configuration per Metric")
    lines.append("")
    lines.append(f"- **Highest Precision**: {best_prec} ({precs[best_prec]:.4f})")
    lines.append(f"- **Highest Recall**: {best_rec} ({recs[best_rec]:.4f})")
    lines.append(f"- **Highest F1 Score**: {best_f1} ({f1s[best_f1]:.4f})")
    lines.append(f"- **Highest Mean IoU**: {best_iou} ({ious[best_iou]:.4f})")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Feature Analysis")
    lines.append("")

    # Rank features by F1
    single_feature_f1 = {
        k: v for k, v in f1s.items() if k != "Full PGRG"
    }
    ranked = sorted(single_feature_f1.items(), key=lambda x: x[1], reverse=True)

    lines.append("### Single-Feature Ranking (by F1 Score)")
    lines.append("")
    for rank, (name, val) in enumerate(ranked, 1):
        lines.append(f"{rank}. **{name}**: F1 = {val:.4f}")
    lines.append("")

    full_f1 = f1s.get("Full PGRG", 0.0)
    if ranked and ranked[0][1] > 0:
        best_single = ranked[0]
        if full_f1 > best_single[1]:
            improvement = ((full_f1 - best_single[1]) / best_single[1]) * 100
            lines.append(f"The Full PGRG combination improves F1 by {improvement:.1f}% over the")
            lines.append(f"best single-feature variant ({best_single[0]}).")
        elif full_f1 == best_single[1]:
            lines.append("The Full PGRG combination achieved the same F1 as the best single-feature variant.")
        else:
            decrease = ((best_single[1] - full_f1) / best_single[1]) * 100
            lines.append(f"The Full PGRG combination achieved a lower F1 ({full_f1:.4f}) than")
            lines.append(f"the best single-feature variant {best_single[0]} ({best_single[1]:.4f}),")
            lines.append(f"a decrease of {decrease:.1f}%.")
    else:
        lines.append("All configurations produced F1 = 0.0 on this sample.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Limitations")
    lines.append("")
    lines.append("1. **Sample size**: The ablation study uses 60 images (10 per class),")
    lines.append("   not the full 1,800-image dataset. Results may differ on the full set.")
    lines.append("2. **Box-level evaluation**: Metrics are computed from bounding-box")
    lines.append("   overlap, not pixel-level segmentation accuracy.")
    lines.append("3. **Single weight configuration per variant**: Each single-feature")
    lines.append("   variant uses weight = 1.0. Alternative weight magnitudes were not tested.")
    lines.append("4. **Runtime variability**: Runtime measurements are single-run per")
    lines.append("   image; they may vary with system load.")
    lines.append("5. **No hyperparameter tuning**: The seed threshold, similarity threshold,")
    lines.append("   and minimum area were held constant across all variants. Performance")
    lines.append("   of individual features might improve with tuned parameters.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Conclusion")
    lines.append("")
    if ranked and ranked[0][1] > 0:
        lines.append(f"Among single-feature variants, **{ranked[0][0]}** achieved the highest")
        lines.append(f"F1 score ({ranked[0][1]:.4f}), suggesting it is the most individually")
        lines.append("informative feature for defect detection with the PGRG algorithm.")
        if full_f1 >= ranked[0][1]:
            lines.append(f"The Full PGRG combination (F1 = {full_f1:.4f}) matches or exceeds")
            lines.append("all single-feature variants, supporting the design decision to")
            lines.append("combine all three features in the priority map.")
        else:
            lines.append(f"However, the Full PGRG combination (F1 = {full_f1:.4f}) did not")
            lines.append(f"outperform {ranked[0][0]} on this sample, suggesting that feature")
            lines.append("weighting may benefit from further optimization.")
    else:
        lines.append("All variants produced zero F1 on this sample. Further investigation is needed.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("# From the project root directory:")
    lines.append("python experiments/run_ablation_study.py")
    lines.append("```")
    lines.append("")
    lines.append("Output files:")
    lines.append("- `results/ablation/ablation_results.csv` — per-image results")
    lines.append("- `results/ablation/ablation_summary.csv` — aggregate results")
    lines.append("- `results/ablation/ablation_comparison.png` — comparison charts")
    lines.append("- `results/ablation/ablation_report.md` — this report")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Saved ablation report: {report_path}")


# ── Console summary ───────────────────────────────────────────────
def print_console_summary(summary_rows):
    """Print a summary table to the console."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY — RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"{'Configuration':<20s} | {'TP':>4s} | {'FP':>5s} | {'FN':>4s} | "
        f"{'Prec':>7s} | {'Recall':>6s} | {'F1':>7s} | {'IoU':>7s} | {'RT(ms)':>8s}"
    )
    print("-" * 70)
    for r in summary_rows:
        print(
            f"{r['config']:<20s} | {r['TP']:>4d} | {r['FP']:>5d} | {r['FN']:>4d} | "
            f"{r['precision']:>7s} | {r['recall']:>6s} | {r['f1']:>7s} | "
            f"{r['mean_iou']:>7s} | {r['mean_runtime_ms']:>8s}"
        )
    print("-" * 70)
    print("\nNOTE: These are box-level localization metrics, NOT pixel-level segmentation.")
    print("=" * 70)


# ── Verification ──────────────────────────────────────────────────
def run_verification_checks(samples, detailed_rows, summary_rows):
    """Run post-experiment verification checks."""
    print("\n" + "-" * 70)
    print("VERIFICATION CHECKS")
    print("-" * 70)

    # V1: All 60 images have matching XML
    v1 = all(os.path.exists(s["xml_path"]) for s in samples)
    print(f"[1] All {len(samples)} images have XML annotations     : {'PASS' if v1 else 'FAIL'}")

    # V2: All 4 configs evaluated on same images
    images_per_config = {}
    for row in detailed_rows:
        cfg = row["config"]
        if cfg not in images_per_config:
            images_per_config[cfg] = set()
        images_per_config[cfg].add(row["image"])

    all_same = len(images_per_config) == 4
    if all_same:
        first_set = list(images_per_config.values())[0]
        all_same = all(s == first_set for s in images_per_config.values())
    print(f"[2] All 4 configs evaluated on same images   : {'PASS' if all_same else 'FAIL'}")

    # V3: Full PGRG uses default weights (None)
    full_pgrg_weights = None
    for name, weights in ABLATION_CONFIGS:
        if name == "Full PGRG":
            full_pgrg_weights = weights
    v3 = full_pgrg_weights is None
    print(f"[3] Full PGRG uses default weights (None)    : {'PASS' if v3 else 'FAIL'}")

    # V4: Detailed row count matches expected
    expected_rows = len(samples) * len(ABLATION_CONFIGS)
    v4 = len(detailed_rows) == expected_rows
    print(f"[4] Detailed CSV has {expected_rows} rows               : {'PASS' if v4 else 'FAIL'}")

    # V5: Summary TP+FP+FN totals match detailed rows
    v5 = True
    for sr in summary_rows:
        cfg = sr["config"]
        detail_tp = sum(int(r["tp"]) for r in detailed_rows if r["config"] == cfg)
        detail_fp = sum(int(r["fp"]) for r in detailed_rows if r["config"] == cfg)
        detail_fn = sum(int(r["fn"]) for r in detailed_rows if r["config"] == cfg)
        if detail_tp != sr["TP"] or detail_fp != sr["FP"] or detail_fn != sr["FN"]:
            v5 = False
            print(f"  [MISMATCH] {cfg}: summary TP/FP/FN != sum of detailed rows")
    print(f"[5] Summary TP/FP/FN totals match details    : {'PASS' if v5 else 'FAIL'}")

    # V6: 10 images per class
    v6 = True
    for cls in CLASSES:
        count = sum(1 for s in samples if s["class"] == cls)
        if count != IMAGES_PER_CLASS:
            v6 = False
            print(f"  [MISMATCH] {cls}: {count} images (expected {IMAGES_PER_CLASS})")
    print(f"[6] Exactly {IMAGES_PER_CLASS} images per class              : {'PASS' if v6 else 'FAIL'}")

    all_pass = v1 and all_same and v3 and v4 and v5 and v6
    print("-" * 70)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("-" * 70)


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    run_ablation_study()
