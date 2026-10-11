"""
preprocessing.py — Image Preprocessing for Steel Defect Detection

Phase 1 implementation:
  - load_image()       : Read a grayscale steel defect image using OpenCV
  - gaussian_blur()    : Apply Gaussian blur using a 5x5 kernel
  - normalize_image()  : Normalize pixel intensity to 0-255 using OpenCV normalization
  - preprocess_image() : Pipeline returning (original, grayscale, blurred, normalized)
"""

import os
import cv2
import numpy as np


def load_image(image_path):
    """
    Read a grayscale steel defect image using OpenCV.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If OpenCV cannot decode or read the image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read or decode image at: {image_path}")

    return image


def gaussian_blur(image, kernel_size=(5, 5)):
    """
    Apply Gaussian blur to reduce high-frequency noise.
    Uses a 5x5 kernel by default.

    Args:
        image (numpy.ndarray): Input image (grayscale or color).
        kernel_size (tuple or int): Size of the Gaussian kernel. Default is (5, 5).

    Returns:
        numpy.ndarray: Blurred image with the same shape and dtype.
    """
    if isinstance(kernel_size, int):
        ksize = (kernel_size, kernel_size)
    else:
        ksize = kernel_size

    blurred = cv2.GaussianBlur(image, ksize, 0)
    return blurred


def normalize_image(image):
    """
    Normalize pixel intensity to the range 0-255 using OpenCV normalization.

    Stretches pixel intensities so the minimum becomes 0 and the maximum becomes 255.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Normalized image with values in [0, 255], dtype uint8.
    """
    normalized = cv2.normalize(
        image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )
    return normalized


def preprocess_image(image_path):
    """
    Preprocess a steel defect image through the complete Phase 1 pipeline:
      1. Load the image.
      2. Convert to grayscale if necessary.
      3. Apply Gaussian blur (5x5 kernel).
      4. Normalize pixel intensity to 0-255.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        tuple: (original, grayscale, blurred, normalized)
            - original   : Raw loaded image as a NumPy array.
            - grayscale  : Grayscale single-channel image.
            - blurred    : Gaussian blurred image.
            - normalized : Intensity normalized image.
    """
    # 1. Load the image
    original = load_image(image_path)

    # 2. Convert to grayscale if necessary
    if original.ndim == 3 and original.shape[2] == 3:
        grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    elif original.ndim == 3 and original.shape[2] == 4:
        grayscale = cv2.cvtColor(original, cv2.COLOR_BGRA2GRAY)
    elif original.ndim == 2:
        grayscale = original.copy()
    else:
        grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 3. Apply Gaussian blur (5x5 kernel)
    blurred = gaussian_blur(grayscale, kernel_size=(5, 5))

    # 4. Normalize pixel intensity to [0, 255]
    normalized = normalize_image(blurred)

    return original, grayscale, blurred, normalized
