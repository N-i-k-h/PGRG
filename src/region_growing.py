"""
region_growing.py — Standard Region Growing (Baseline 1)

Conventional region growing segmentation for steel surface defect detection.
Serves as Baseline 1 to compare against Priority-Guided Region Growing (PGRG).

Key Characteristics:
  - Uses preprocessed normalized grayscale image.
  - 8-neighbour connectivity.
  - Seeds selected via intensity threshold on normalized image.
  - Conventional FIFO queue (deque) order — NO priority map, NO max-heap.
  - Region acceptance: abs(neighbour_intensity - region_mean) <= similarity_threshold.
  - Tracks visited pixels and outputs binary defect mask (0 = background, 255 = defect).
"""

from collections import deque
import numpy as np


# 8-neighbourhood offsets: (-1,-1) to (1,1) excluding (0,0)
NEIGHBORS_8 = [
    (-1, -1), (-1,  0), (-1,  1),
    ( 0, -1),           ( 0,  1),
    ( 1, -1), ( 1,  0), ( 1,  1),
]


def select_seeds(image, seed_threshold=0.45):
    """
    Select seed pixels from preprocessed grayscale image where normalized intensity
    meets or exceeds seed_threshold.

    Args:
        image (numpy.ndarray): 2D grayscale or normalized image (H, W).
        seed_threshold (float): Minimum normalized intensity in [0, 1] to qualify as seed.
            Default is 0.45.

    Returns:
        list of tuple: List of (y, x) coordinates for seed pixels.
    """
    if image.ndim == 3:
        image = image[:, :, 0]

    # Normalize image to [0.0, 1.0] if in [0, 255]
    if image.max() > 1.0:
        norm_img = image.astype(np.float32) / 255.0
    else:
        norm_img = image.astype(np.float32)

    ys, xs = np.where(norm_img >= seed_threshold)
    seeds = [(int(y), int(x)) for y, x in zip(ys, xs)]
    return seeds


def standard_region_growing(
    image,
    seed_threshold=0.45,
    similarity_threshold=20,
    seeds=None,
    return_details=False,
    return_stats=False,
    max_pops=None
):
    """
    Standard Region Growing (SRG) baseline segmentation algorithm.

    Grows defect regions from initial seed pixels using a conventional FIFO queue
    and intensity similarity criterion. Does NOT use any priority map or max-heap.

    Algorithm:
      1. Select initial seeds where normalized intensity >= seed_threshold (if seeds not provided).
      2. Initialize FIFO queue with seed coordinates.
      3. Mark seeds in visited array and region mask (value 255).
      4. While queue is not empty:
         a. Pop pixel (y, x) from the front of FIFO queue.
         b. Compute current region mean intensity.
         c. For each of 8 neighbours:
            - Skip if out of bounds or already visited.
            - If abs(neighbour_intensity - region_mean) <= similarity_threshold:
              * Mark visited = True.
              * Add to region mask (255).
              * Update running region intensity sum and count.
              * Push neighbour to back of FIFO queue.
      5. Return binary region mask (values 0 and 255).

    Args:
        image (numpy.ndarray): Preprocessed normalized grayscale steel defect image (H, W).
        seed_threshold (float): Intensity threshold for seed selection in [0, 1]. Default is 0.45.
        similarity_threshold (float): Maximum absolute difference between neighbour intensity
            and running region mean. Default is 20.
        seeds (list of tuple, optional): Explicit seed coordinates [(y, x), ...].
            If None, seeds are selected automatically using seed_threshold.
        return_details (bool, optional): If True, returns (region_mask, visited, processed_pixels).
            If False, returns region_mask. Default is False.
        return_stats (bool, optional): If True, returns (region_mask, visited, stats_dict).
            Default is False.
        max_pops (int, optional): Maximum number of queue pop operations allowed.
            If None, executes until queue is empty (unrestricted). Default is None.

    Returns:
        numpy.ndarray or tuple:
            - If return_stats is True: tuple (region_mask, visited, stats_dict).
            - If return_details is True: tuple (region_mask, visited, processed_pixels).
            - If return_details is False and return_stats is False: region_mask (uint8 array with values {0, 255}).
    """
    # Ensure 2D image array
    if image.ndim == 3:
        image = image[:, :, 0]
    img_data = image.astype(np.float32)
    h, w = img_data.shape

    # 1. Select seed pixels
    if seeds is None:
        seed_list = select_seeds(img_data, seed_threshold=seed_threshold)
    else:
        seed_list = list(seeds)

    # 2. Initialize data structures
    visited = np.zeros((h, w), dtype=bool)
    region_mask = np.zeros((h, w), dtype=np.uint8)
    queue = deque()

    # Initialize running sum and count for region mean
    region_sum = 0.0
    for y, x in seed_list:
        if 0 <= y < h and 0 <= x < w and not visited[y, x]:
            visited[y, x] = True
            region_mask[y, x] = 255
            region_sum += float(img_data[y, x])
            queue.append((y, x))

    region_count = len(queue)
    number_of_processed_pixels = 0
    neighbour_checks = 0
    similarity_checks = 0

    # 3. FIFO Region Growing
    while queue:
        if max_pops is not None and number_of_processed_pixels >= max_pops:
            break
        # a. Pop from front of FIFO queue
        y, x = queue.popleft()
        number_of_processed_pixels += 1

        # b. Compute current region mean intensity
        region_mean = region_sum / region_count if region_count > 0 else 0.0

        # c. Examine all 8 neighbours
        for dy, dx in NEIGHBORS_8:
            neighbour_checks += 1
            ny, nx = y + dy, x + dx

            # Boundary check
            if not (0 <= ny < h and 0 <= nx < w):
                continue

            # Visited check
            if visited[ny, nx]:
                continue

            # Intensity similarity check
            neighbour_intensity = float(img_data[ny, nx])
            intensity_diff = abs(neighbour_intensity - region_mean)
            similarity_checks += 1

            if intensity_diff <= similarity_threshold:
                visited[ny, nx] = True
                region_mask[ny, nx] = 255
                region_sum += neighbour_intensity
                region_count += 1
                queue.append((ny, nx))

    if return_stats:
        stats = {
            "pops": number_of_processed_pixels,
            "unique_visited": int(np.sum(visited)),
            "neighbour_checks": neighbour_checks,
            "similarity_checks": similarity_checks,
            "seeds_count": len(seed_list),
            "total_pixels": h * w,
        }
        return region_mask, visited, stats

    if return_details:
        return region_mask, visited, number_of_processed_pixels

    return region_mask
