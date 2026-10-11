"""
otsu.py — Otsu Thresholding + Morphological Post-Processing (Baseline 2)

Conventional segmentation baseline for steel surface defect detection.
Applies Otsu's global binarization followed by morphological opening and closing
to remove noise and fill small discontinuities.

Key Characteristics:
  - Uses preprocessed normalized grayscale image.
  - Computes optimal threshold via cv2.threshold(..., cv2.THRESH_BINARY + cv2.THRESH_OTSU).
  - Cleans binary mask using morphological opening (cv2.MORPH_OPEN) with a (3, 3) elliptical kernel.
  - Closes small gaps using morphological closing (cv2.MORPH_CLOSE) with the same structuring element.
  - Does NOT use priority map, max-heap, Union-Find, or region growing.
  - Returns (otsu_threshold, binary_mask, cleaned_mask).
"""

import cv2
import numpy as np


def otsu_morphology(image):
    """
    Segment defects using Otsu thresholding followed by morphological operations.

    Algorithm:
      1. Ensure image is single-channel uint8.
      2. Compute global Otsu threshold and generate raw binary mask.
      3. Create a 3x3 elliptical structuring element.
      4. Apply morphological opening to eliminate isolated noise.
      5. Apply morphological closing to bridge narrow gaps within defect regions.
      6. Return threshold value, raw binary mask, and cleaned binary mask.

    Args:
        image (numpy.ndarray): Preprocessed normalized grayscale steel defect image (H, W).

    Returns:
        tuple: (otsu_threshold, binary_mask, cleaned_mask)
            - otsu_threshold (float): Automatically determined Otsu threshold value.
            - binary_mask (numpy.ndarray): Raw thresholded binary mask (uint8, {0, 255}).
            - cleaned_mask (numpy.ndarray): Morphologically cleaned binary mask (uint8, {0, 255}).
    """
    # 1. Ensure 2D uint8 image
    if image.ndim == 3:
        image = image[:, :, 0]

    if image.dtype != np.uint8:
        img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image

    # 2. Otsu thresholding
    otsu_threshold, binary_mask = cv2.threshold(
        img_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3. Morphological structuring element (3x3 ellipse)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    # 4. Morphological opening to remove small noise
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # 5. Morphological closing to fill small gaps
    cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

    return float(otsu_threshold), binary_mask, cleaned_mask
