"""
pixel_metrics.py — Pixel-Level Segmentation Evaluation Metrics

Provides functions to compute pixel-level segmentation performance metrics
comparing predicted binary defect masks against genuine ground-truth masks:
  - pixel_tp, pixel_fp, pixel_fn, pixel_tn (Confusion Matrix)
  - Pixel Precision
  - Pixel Recall
  - Pixel F1 / Dice Score
  - Pixel IoU / Jaccard Index
  - Summary aggregation (macro and micro averages)
"""

import numpy as np


def compute_pixel_confusion_matrix(pred_mask, gt_mask):
    """
    Compute pixel-level confusion matrix entries (TP, FP, FN, TN).

    Args:
        pred_mask (np.ndarray): Binary prediction mask (boolean or integer, shape HxW).
        gt_mask (np.ndarray): Binary ground-truth mask (boolean or integer, shape HxW).

    Returns:
        tuple: (tp, fp, fn, tn) as integers.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred_mask {pred_mask.shape} vs gt_mask {gt_mask.shape}")

    p = (pred_mask > 0).astype(bool)
    g = (gt_mask > 0).astype(bool)

    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())

    return tp, fp, fn, tn


def calculate_pixel_metrics(pred_mask, gt_mask):
    """
    Calculate pixel-level precision, recall, F1/Dice score, and IoU/Jaccard index.

    Edge-case Handling:
      - If both pred and gt contain zero defect pixels:
          Dice = 1.0, IoU = 1.0, Precision = 1.0, Recall = 1.0 (perfect background agreement).
      - If pred has defect pixels but gt has zero defect pixels:
          Precision = 0.0, Recall = 1.0, F1 = 0.0, IoU = 0.0.
      - If pred has zero defect pixels but gt has defect pixels:
          Precision = 1.0, Recall = 0.0, F1 = 0.0, IoU = 0.0.

    Args:
        pred_mask (np.ndarray): Predicted binary mask (HxW).
        gt_mask (np.ndarray): Ground truth binary mask (HxW).

    Returns:
        dict: Dictionary containing:
            - tp, fp, fn, tn
            - precision
            - recall
            - f1 (Dice coefficient)
            - iou (Jaccard index)
            - pred_defect_pixels
            - gt_defect_pixels
            - total_pixels
    """
    tp, fp, fn, tn = compute_pixel_confusion_matrix(pred_mask, gt_mask)
    total_pixels = pred_mask.size
    pred_defect_pixels = tp + fp
    gt_defect_pixels = tp + fn

    # Empty mask edge cases
    if pred_defect_pixels == 0 and gt_defect_pixels == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
        iou = 1.0
    elif pred_defect_pixels > 0 and gt_defect_pixels == 0:
        precision = 0.0
        recall = 1.0
        f1 = 0.0
        iou = 0.0
    elif pred_defect_pixels == 0 and gt_defect_pixels > 0:
        precision = 1.0
        recall = 0.0
        f1 = 0.0
        iou = 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "pred_defect_pixels": pred_defect_pixels,
        "gt_defect_pixels": gt_defect_pixels,
        "total_pixels": total_pixels,
    }


def aggregate_pixel_metrics(per_image_results):
    """
    Compute macro-averaged and micro-averaged pixel metrics across multiple images.

    Args:
        per_image_results (list of dict): List of dicts returned by calculate_pixel_metrics.

    Returns:
        dict: Aggregated summary with both macro (mean of per-image) and micro (pooled confusion matrix).
    """
    if not per_image_results:
        return {
            "count": 0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "macro_iou": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "micro_iou": 0.0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
            "total_tn": 0,
        }

    n = len(per_image_results)
    macro_prec = sum(r["precision"] for r in per_image_results) / n
    macro_rec = sum(r["recall"] for r in per_image_results) / n
    macro_f1 = sum(r["f1"] for r in per_image_results) / n
    macro_iou = sum(r["iou"] for r in per_image_results) / n

    total_tp = sum(r["tp"] for r in per_image_results)
    total_fp = sum(r["fp"] for r in per_image_results)
    total_fn = sum(r["fn"] for r in per_image_results)
    total_tn = sum(r["tn"] for r in per_image_results)

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0
    micro_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    return {
        "count": n,
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "macro_iou": float(macro_iou),
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        "micro_f1": float(micro_f1),
        "micro_iou": float(micro_iou),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
    }
