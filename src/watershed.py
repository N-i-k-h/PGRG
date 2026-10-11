"""
watershed.py — Watershed Segmentation (Baseline 3)

Conventional segmentation baseline for steel surface defect detection.
Applies marker-based watershed segmentation using OpenCV cv2.watershed().

Key Characteristics:
  - Uses preprocessed normalized grayscale image.
  - Builds binary/edge representation via statistical contrast thresholding.
  - Generates markers via morphological operations and distance transform.
  - Applies cv2.watershed() to partition the image into distinct regions.
  - Extracts final binary defect mask (0 = background, 255 = defect).
  - Does NOT use priority map, max-heap, PGRG, Union-Find, Standard Region Growing,
    or Otsu thresholding as segmentation method.
  - Returns (markers, defect_mask).
"""

import cv2
import numpy as np


def watershed_segmentation(image, return_markers=True):
    """
    Segment defects using marker-controlled watershed segmentation.

    Algorithm:
      1. Ensure image is single-channel uint8.
      2. Threshold based on statistical contrast (mean + std) to identify candidate regions.
      3. Clean noise using morphological opening with a rectangular kernel.
      4. Determine sure background by dilating the opened candidate regions.
      5. Compute distance transform on opened regions and threshold to find sure foreground seeds.
      6. Identify unknown border areas via sure_bg - sure_fg.
      7. Label connected components on sure foreground as distinct markers (offset by +1).
      8. Assign 0 to unknown areas so the watershed algorithm resolves the boundaries.
      9. Convert image to 3-channel BGR and execute cv2.watershed(img_bgr, markers).
      10. Generate final binary defect mask: pixels with marker label > 1 are defect (255),
          while background (1) and watershed boundaries (-1) are background (0).

    Args:
        image (numpy.ndarray): Preprocessed normalized grayscale steel defect image (H, W).
        return_markers (bool, optional): If True, returns (markers, defect_mask).
            If False, returns defect_mask. Default is True.

    Returns:
        tuple or numpy.ndarray:
            - If return_markers is True: (markers, defect_mask)
                * markers (numpy.ndarray): int32 label map produced by cv2.watershed (-1, 1, 2, ...).
                * defect_mask (numpy.ndarray): Binary uint8 mask (0 or 255) of detected defects.
            - If return_markers is False: defect_mask.
    """
    # 1. Ensure 2D uint8 image
    if image.ndim == 3:
        image = image[:, :, 0]

    if image.dtype != np.uint8:
        img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image

    h, w = img_uint8.shape

    # 2. Binary candidate representation via statistical contrast thresholding
    # (mean + std identifies high-contrast defect candidates without using Otsu)
    thresh_val = float(img_uint8.mean() + img_uint8.std())
    _, binary = cv2.threshold(img_uint8, int(thresh_val), 255, cv2.THRESH_BINARY)

    # 3. Noise removal via morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. Sure background area via dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 5. Sure foreground area via Euclidean distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    if max_dist > 0:
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * max_dist, 255, cv2.THRESH_BINARY)
    else:
        sure_fg = np.zeros((h, w), dtype=np.uint8)
    sure_fg = np.uint8(sure_fg)

    # 6. Unknown region between sure background and sure foreground
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. Marker labelling using connected components
    num_markers, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all labels so that sure background is 1 instead of 0
    markers = markers + 1

    # 8. Mark unknown region as 0 for watershed to resolve
    markers[unknown == 255] = 0

    # 9. Convert to 3-channel BGR as required by cv2.watershed
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    # Execute OpenCV watershed
    markers = cv2.watershed(img_bgr, markers)

    # 10. Generate final binary defect mask (0 = background, 255 = defect)
    # Background is label 1; watershed boundary lines are label -1.
    defect_mask = np.zeros((h, w), dtype=np.uint8)
    defect_mask[markers > 1] = 255

    if return_markers:
        return markers, defect_mask

    return defect_mask
