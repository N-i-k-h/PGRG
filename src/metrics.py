"""
metrics.py — Ground-Truth Bounding Box Loader and Detection Metrics

Evaluation module for steel defect detection:
  - load_ground_truth_boxes()     : Parse PASCAL VOC XML ground-truth annotations
  - mask_to_boxes()               : Extract bounding boxes from connected components of binary mask
  - compute_box_iou()             : Standard intersection-over-union between two boxes
  - match_boxes()                 : Greedy one-to-one matching between predicted and GT boxes
  - calculate_detection_metrics() : Precision, Recall, F1, and Mean Matched IoU
  - measure_runtime()             : High-precision execution timing helper via time.perf_counter()
"""

import time
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def load_ground_truth_boxes(xml_path):
    """
    Parse a PASCAL VOC XML annotation file and return all ground-truth bounding boxes.

    Args:
        xml_path (str): Path to the XML annotation file.

    Returns:
        list of dict: List of bounding boxes, each formatted as:
            {
                "class": <str>,
                "xmin": <int>,
                "ymin": <int>,
                "xmax": <int>,
                "ymax": <int>
            }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            boxes.append({
                "class": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })

    return boxes


def mask_to_boxes(binary_mask):
    """
    Extract bounding boxes for all connected defect components in a binary mask.

    Uses OpenCV cv2.connectedComponentsWithStats to find foreground regions.
    Ignores background (label 0) and zero-area components.

    Args:
        binary_mask (numpy.ndarray): 2D binary defect mask ({0, 255} or boolean).

    Returns:
        list of dict: List of bounding boxes:
            [
                {
                    "xmin": <int>,
                    "ymin": <int>,
                    "xmax": <int>,
                    "ymax": <int>
                },
                ...
            ]
    """
    mask_uint8 = (binary_mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)

    boxes = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if w > 0 and h > 0 and area > 0:
            boxes.append({
                "xmin": x,
                "ymin": y,
                "xmax": x + w,
                "ymax": y + h
            })

    return boxes


def compute_box_iou(box_a, box_b):
    """
    Calculate the standard Intersection over Union (IoU) between two bounding boxes.

    IoU = intersection_area / union_area

    Args:
        box_a (dict): Bounding box with keys 'xmin', 'ymin', 'xmax', 'ymax'.
        box_b (dict): Bounding box with keys 'xmin', 'ymin', 'xmax', 'ymax'.

    Returns:
        float: IoU value in [0.0, 1.0]. Returns 0.0 for non-overlapping boxes.
    """
    # Intersection coordinates
    ixmin = max(box_a["xmin"], box_b["xmin"])
    iymin = max(box_a["ymin"], box_b["ymin"])
    ixmax = min(box_a["xmax"], box_b["xmax"])
    iymax = min(box_a["ymax"], box_b["ymax"])

    iw = max(0, ixmax - ixmin)
    ih = max(0, iymax - iymin)
    intersection = iw * ih

    if intersection <= 0:
        return 0.0

    # Areas of individual boxes
    area_a = max(0, box_a["xmax"] - box_a["xmin"]) * max(0, box_a["ymax"] - box_a["ymin"])
    area_b = max(0, box_b["xmax"] - box_b["xmin"]) * max(0, box_b["ymax"] - box_b["ymin"])

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def match_boxes(predicted_boxes, ground_truth_boxes, iou_threshold=0.5):
    """
    Perform greedy one-to-one matching between predicted and ground-truth boxes.

    Each predicted box can match at most one ground-truth box with IoU >= iou_threshold.
    Matches are resolved greedily in descending order of IoU score.

    Args:
        predicted_boxes (list of dict): List of predicted boxes.
        ground_truth_boxes (list of dict): List of ground-truth boxes.
        iou_threshold (float): Minimum IoU threshold to consider a match. Default is 0.5.

    Returns:
        tuple: (true_positives, false_positives, false_negatives, matched_pairs)
            - true_positives (int): Number of successfully matched predictions.
            - false_positives (int): Number of unmatched predictions.
            - false_negatives (int): Number of unmatched ground-truth boxes.
            - matched_pairs (list of tuple): List of (pred_box, gt_box, iou).
    """
    candidates = []
    for i, pred in enumerate(predicted_boxes):
        for j, gt in enumerate(ground_truth_boxes):
            iou = compute_box_iou(pred, gt)
            if iou >= iou_threshold:
                candidates.append((iou, i, j))

    # Sort descending by IoU
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_preds = set()
    matched_gts = set()
    matched_pairs = []

    for iou, i, j in candidates:
        if i not in matched_preds and j not in matched_gts:
            matched_preds.add(i)
            matched_gts.add(j)
            matched_pairs.append((predicted_boxes[i], ground_truth_boxes[j], float(iou)))

    tp = len(matched_pairs)
    fp = len(predicted_boxes) - tp
    fn = len(ground_truth_boxes) - tp

    return tp, fp, fn, matched_pairs


def calculate_detection_metrics(true_positives, false_positives, false_negatives, matched_pairs=None):
    """
    Compute Precision, Recall, F1 score, and Mean Matched-Box IoU.

    Handles zero denominators safely.

    Args:
        true_positives (int): Count of true positive detections.
        false_positives (int): Count of false positive detections.
        false_negatives (int): Count of false negative detections.
        matched_pairs (list of tuple, optional): List of (pred, gt, iou) tuples.

    Returns:
        dict: Detection metrics:
            {
                "precision": float,
                "recall": float,
                "f1": float,
                "mean_iou": float
            }
    """
    tp = float(true_positives)
    fp = float(false_positives)
    fn = float(false_negatives)

    # Precision = TP / (TP + FP)
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # Recall = TP / (TP + FN)
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # F1 = 2 * Precision * Recall / (Precision + Recall)
    if (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # Mean IoU of matched boxes
    if matched_pairs and len(matched_pairs) > 0:
        mean_iou = float(sum(p[2] for p in matched_pairs) / len(matched_pairs))
    else:
        mean_iou = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou
    }


def measure_runtime(func, *args, **kwargs):
    """
    Measure execution time of a callable in milliseconds using time.perf_counter().

    Excludes any visualization, printing, or file-saving overhead.

    Args:
        func (callable): Function to benchmark.
        *args: Positional arguments to pass to func.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        tuple: (result, runtime_ms)
            - result: Return value of func(*args, **kwargs).
            - runtime_ms (float): Execution duration in milliseconds.
    """
    t_start = time.perf_counter()
    result = func(*args, **kwargs)
    t_end = time.perf_counter()
    runtime_ms = (t_end - t_start) * 1000.0
    return result, runtime_ms
