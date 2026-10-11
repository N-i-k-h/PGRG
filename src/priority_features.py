"""
priority_features.py — Priority Feature Extraction for PGRG Steel Defect Detection

Phase 2 implementation:
  - compute_local_mean()             : Mean intensity in a local neighborhood (7x7)
  - compute_intensity_deviation()    : Absolute deviation |I(x, y) - local_mean(x, y)|
  - compute_sobel_gradient()         : Gradient magnitude from Sobel Gx, Gy normalized to [0, 255]
  - compute_local_texture_variance() : Texture variance = mean(I^2) - mean(I)^2 normalized to [0, 255]
  - compute_priority_features()      : Pipeline returning all 4 features
"""

import cv2
import numpy as np


def compute_local_mean(image, kernel_size=(7, 7)):
    """
    Calculate the local mean intensity using a neighborhood window (default 7x7).

    Args:
        image (numpy.ndarray): Grayscale or normalized input image.
        kernel_size (tuple or int): Size of the neighborhood kernel, default (7, 7).

    Returns:
        numpy.ndarray: Local mean image of the same shape and type.
    """
    if isinstance(kernel_size, int):
        ksize = (kernel_size, kernel_size)
    else:
        ksize = kernel_size

    return cv2.blur(image, ksize)


def compute_intensity_deviation(image, local_mean):
    """
    Calculate the intensity deviation: |I(x, y) - local_mean(x, y)|.

    Args:
        image (numpy.ndarray): Input image.
        local_mean (numpy.ndarray): Local mean image.

    Returns:
        numpy.ndarray: Deviation image.
    """
    if image.dtype == np.uint8 and local_mean.dtype == np.uint8:
        return cv2.absdiff(image, local_mean)

    deviation = np.abs(image.astype(np.float32) - local_mean.astype(np.float32))
    return deviation


def compute_sobel_gradient(image):
    """
    Calculate the Sobel gradient magnitude:
        Gx = Sobel_x(image)
        Gy = Sobel_y(image)
        magnitude = sqrt(Gx^2 + Gy^2)
    Normalized to [0, 255] for visualization.

    Args:
        image (numpy.ndarray): Grayscale or normalized input image.

    Returns:
        numpy.ndarray: Gradient magnitude normalized to [0, 255] (uint8).
    """
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(grad_x, grad_y)

    normalized_magnitude = cv2.normalize(
        magnitude,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )
    return normalized_magnitude


def compute_local_texture_variance(image, kernel_size=(7, 7)):
    """
    Calculate the local texture variance:
        variance = mean(I^2) - mean(I)^2
    Prevents negative values due to floating-point imprecision and normalizes
    to [0, 255] for visualization.

    Args:
        image (numpy.ndarray): Grayscale or normalized input image.
        kernel_size (tuple or int): Size of the neighborhood kernel, default (7, 7).

    Returns:
        numpy.ndarray: Local texture variance normalized to [0, 255] (uint8).
    """
    if isinstance(kernel_size, int):
        ksize = (kernel_size, kernel_size)
    else:
        ksize = kernel_size

    img_float = image.astype(np.float32)

    # Local mean of intensity: mean(I)
    mean_i = cv2.blur(img_float, ksize)

    # Local mean of squared intensity: mean(I^2)
    mean_i_sq = cv2.blur(img_float ** 2, ksize)

    # Variance: mean(I^2) - mean(I)^2
    variance = mean_i_sq - (mean_i ** 2)

    # Prevent small numerical negative values caused by floating-point calculations
    variance = np.maximum(variance, 0.0)

    # Normalize to [0, 255] for visualization
    normalized_variance = cv2.normalize(
        variance,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )
    return normalized_variance


def compute_priority_features(image):
    """
    Compute all four priority features:
      1. Local mean
      2. Intensity deviation
      3. Sobel gradient magnitude
      4. Local texture variance

    Args:
        image (numpy.ndarray): Normalized steel surface defect image.

    Returns:
        tuple: (local_mean, intensity_deviation, gradient_magnitude, texture_variance)
    """
    local_mean = compute_local_mean(image, kernel_size=(7, 7))
    intensity_deviation = compute_intensity_deviation(image, local_mean)
    gradient_magnitude = compute_sobel_gradient(image)
    texture_variance = compute_local_texture_variance(image, kernel_size=(7, 7))

    return local_mean, intensity_deviation, gradient_magnitude, texture_variance
