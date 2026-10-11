"""
priority_map.py — Priority Map Generation for Steel Defect Detection

Phase 3 implementation:
  - normalize_feature()      : Normalize feature array to [0.0, 1.0] with div-by-zero safety
  - compute_priority_map()   : Weighted combination of deviation, gradient, texture variance
  - priority_map_uint8()     : Convert [0, 1] priority map to [0, 255] uint8 for visualization
"""

import numpy as np


def normalize_feature(feature):
    """
    Convert feature array to float32 and normalize pixel intensities to [0, 1].
    Handles constant features safely without division-by-zero.

    Args:
        feature (numpy.ndarray): Input feature array.

    Returns:
        numpy.ndarray: Normalized feature in [0.0, 1.0], dtype float32.
    """
    f = feature.astype(np.float32)
    f_min = float(f.min())
    f_max = float(f.max())

    if f_max > f_min:
        return (f - f_min) / (f_max - f_min)
    else:
        # Constant feature map: return zeros safely
        return np.zeros_like(f, dtype=np.float32)


def compute_priority_map(
    intensity_deviation,
    gradient_magnitude,
    texture_variance,
    weights=None
):
    """
    Compute the priority map by combining three defect-related features:
      1. Normalized Intensity Deviation
      2. Normalized Sobel Gradient Magnitude
      3. Normalized Local Texture Variance

    Note: Local mean is NOT used directly here as it is already captured
    via intensity deviation.

    Args:
        intensity_deviation (numpy.ndarray): Intensity deviation feature map.
        gradient_magnitude (numpy.ndarray): Sobel gradient magnitude feature map.
        texture_variance (numpy.ndarray): Local texture variance feature map.
        weights (tuple, list, or dict, optional): Weights for (deviation, gradient, texture).
            Defaults to equal weights (1/3, 1/3, 1/3).

    Returns:
        numpy.ndarray: Priority map with values clipped to [0, 1], dtype float32.
    """
    # 1. Normalize all three features independently to [0, 1]
    norm_dev = normalize_feature(intensity_deviation)
    norm_grad = normalize_feature(gradient_magnitude)
    norm_tex = normalize_feature(texture_variance)

    # 2. Configurable weights (defaults to 1/3 each)
    if weights is None:
        w_dev, w_grad, w_tex = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    elif isinstance(weights, dict):
        w_dev = float(weights.get("deviation", 1.0 / 3.0))
        w_grad = float(weights.get("gradient", 1.0 / 3.0))
        w_tex = float(weights.get("texture", 1.0 / 3.0))
    elif len(weights) == 3:
        w_dev, w_grad, w_tex = float(weights[0]), float(weights[1]), float(weights[2])
    else:
        raise ValueError("weights must be None, a 3-element sequence, or a dict.")

    # 3. Calculate weighted linear combination
    priority_map = (
        w_dev * norm_dev +
        w_grad * norm_grad +
        w_tex * norm_tex
    )

    # 4. Ensure float32 dtype and strictly clipped to [0, 1]
    priority_map = np.clip(priority_map, 0.0, 1.0).astype(np.float32)

    return priority_map


def priority_map_uint8(priority_map):
    """
    Convert a [0, 1] float32 priority map to [0, 255] uint8 for visualization.

    Args:
        priority_map (numpy.ndarray): Float priority map with values in [0, 1].

    Returns:
        numpy.ndarray: Priority map converted to uint8 in [0, 255].
    """
    clipped = np.clip(priority_map, 0.0, 1.0)
    return (clipped * 255.0).round().astype(np.uint8)
