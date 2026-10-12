"""
pgrg.py — Priority-Guided Region Growing (PGRG)

Core algorithm implementation:
  - select_seeds()                   : Select initial seed pixels where priority >= threshold
  - initialize_max_heap()            : Initialize max-heap with (-priority, y, x) and visited matrix
  - priority_guided_region_growing() : Full Priority-Guided Region Growing algorithm (8-connected)
"""

import heapq
import cv2
import numpy as np

from src.priority_features import compute_priority_features
from src.priority_map import compute_priority_map
from src.union_find import build_region_uf
from src.region_filtering import filter_regions


# 8-neighbourhood offsets: (-1,-1) to (1,1) excluding (0,0)
NEIGHBORS_8 = [
    (-1, -1), (-1,  0), (-1,  1),
    ( 0, -1),           ( 0,  1),
    ( 1, -1), ( 1,  0), ( 1,  1),
]


def select_seeds(priority_map, seed_threshold):
    """
    Select seed pixels whose priority value is greater than or equal to seed_threshold.

    A pixel (y, x) is chosen as a seed if:
        priority_map[y, x] >= seed_threshold

    Args:
        priority_map (numpy.ndarray): 2D float32 priority map in [0, 1].
        seed_threshold (float): Minimum priority value required to be a seed.

    Returns:
        list of tuple: List of (y, x) coordinates for all selected seed pixels.
    """
    ys, xs = np.where(priority_map >= seed_threshold)
    seeds = [(int(y), int(x)) for y, x in zip(ys, xs)]
    return seeds


def initialize_max_heap(priority_map, seeds):
    """
    Create a max-heap containing all selected seed pixels and initialize
    the boolean visited matrix.

    Each heap entry is a tuple:
        (-priority, y, x)
    Negative priority is used because Python's heapq implements a min-heap,
    ensuring that the pixel with the highest priority is popped first.

    Args:
        priority_map (numpy.ndarray): 2D float32 priority map.
        seeds (list of tuple): Seed coordinates [(y, x), ...].

    Returns:
        tuple: (heap, visited)
            - heap (list): Max-heap (heapq list) of (-priority, y, x) entries.
            - visited (numpy.ndarray): 2D boolean array with seeds marked as True.
    """
    visited = np.zeros(priority_map.shape, dtype=bool)
    heap = []

    for y, x in seeds:
        priority = float(priority_map[y, x])
        heap.append((-priority, int(y), int(x)))
        visited[y, x] = True

    # Transform list into a heap in-place in linear time
    heapq.heapify(heap)

    return heap, visited


def priority_guided_region_growing(
    image,
    priority_map,
    seed_threshold=0.45,
    similarity_threshold=20,
    return_stats=False,
    max_pops=None
):
    """
    Priority-Guided Region Growing (PGRG) algorithm for steel surface defect segmentation.

    The algorithm uses the priority map to control the order in which pixels are
    processed. The pixel with the highest priority in the max-heap is popped first,
    guaranteeing that regions grow preferentially along the most prominent defect structures.

    Algorithm:
      1. Select initial seeds where priority_map[y, x] >= seed_threshold.
      2. Initialize max-heap, visited matrix, and binary region mask.
      3. While max-heap is not empty:
         a. Pop pixel (y, x) with the highest priority.
         b. Compute the current region mean intensity.
         c. Examine all 8 neighbouring pixels:
            - Ignore neighbours outside image boundaries or already visited.
            - For unvisited neighbours, calculate:
                intensity_difference = abs(neighbour_intensity - region_mean)
            - If intensity_difference <= similarity_threshold:
                * Mark neighbour as visited.
                * Add neighbour to region_mask.
                * Push neighbour into max-heap using (-priority, ny, nx).
      4. Return region_mask, visited, number_of_processed_pixels (or stats if return_stats=True).

    Args:
        image (numpy.ndarray): Grayscale or normalized steel defect image (H, W).
        priority_map (numpy.ndarray): 2D float32 priority map (H, W) in [0, 1].
        seed_threshold (float): Minimum priority value to qualify as a seed. Default is 0.45.
        similarity_threshold (float): Maximum intensity difference to accept neighbour. Default is 20.
        return_stats (bool, optional): If True, returns (region_mask, visited, stats_dict).
            Default is False.
        max_pops (int, optional): Maximum number of heap pop operations allowed.
            If None, executes until the heap is empty (unrestricted). Default is None.

    Returns:
        tuple:
            - If return_stats is False: (region_mask, visited, number_of_processed_pixels)
            - If return_stats is True: (region_mask, visited, stats_dict)
    """
    # Ensure 2D image array
    if image.ndim == 3:
        image = image[:, :, 0]
    img_data = image.astype(np.float32)
    h, w = image.shape

    # 1. Select seed pixels
    seeds = select_seeds(priority_map, seed_threshold=seed_threshold)

    # 2. Initialize max-heap, visited matrix, and region mask
    heap, visited = initialize_max_heap(priority_map, seeds)
    region_mask = np.zeros((h, w), dtype=np.uint8)

    # Add seeds to region mask and initialize running sum / count for region mean
    region_sum = 0.0
    for y, x in seeds:
        region_mask[y, x] = 1
        region_sum += float(img_data[y, x])
    region_count = len(seeds)

    number_of_processed_pixels = 0
    neighbour_checks = 0
    similarity_checks = 0

    # 4. Process pixels in priority order
    while heap:
        if max_pops is not None and number_of_processed_pixels >= max_pops:
            break
        # a. Pop pixel with the highest priority
        neg_priority, y, x = heapq.heappop(heap)
        number_of_processed_pixels += 1

        # c. Determine the current region mean intensity
        region_mean = region_sum / region_count if region_count > 0 else 0.0

        # d. Examine all 8 neighbouring pixels
        for dy, dx in NEIGHBORS_8:
            neighbour_checks += 1
            ny, nx = y + dy, x + dx

            # e. Ignore neighbours that are outside boundaries or already visited
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if visited[ny, nx]:
                continue

            # f. Calculate intensity difference
            neighbour_intensity = float(img_data[ny, nx])
            intensity_difference = abs(neighbour_intensity - region_mean)
            similarity_checks += 1

            # g. Check similarity threshold
            if intensity_difference <= similarity_threshold:
                visited[ny, nx] = True
                region_mask[ny, nx] = 1
                region_sum += neighbour_intensity
                region_count += 1
                neighbour_priority = float(priority_map[ny, nx])
                heapq.heappush(heap, (-neighbour_priority, ny, nx))

    if return_stats:
        stats = {
            "pops": number_of_processed_pixels,
            "unique_visited": int(np.sum(visited)),
            "neighbour_checks": neighbour_checks,
            "similarity_checks": similarity_checks,
            "seeds_count": len(seeds),
            "total_pixels": h * w,
        }
        return region_mask, visited, stats

    return region_mask, visited, number_of_processed_pixels


# ---------------------------------------------------------------------------
# Verified Algorithm Configurations
# ---------------------------------------------------------------------------

# Verified Improved PGRG Configuration (New Default)
# Original PGRG Configuration (min_area=100, similarity_threshold=25, merge_gap=11)
ORIGINAL_CONFIG = {
    "seed_threshold": 0.45,
    "similarity_threshold": 25,
    "min_area": 100,
    "merge_gap": 11,
    "weights": None,
}
IMPROVED_CONFIG = ORIGINAL_CONFIG



def pgrg_pipeline(
    image,
    seed_threshold=0.45,
    similarity_threshold=25,
    min_area=100,
    merge_gap=11,
    weights=None,
    return_intermediate=False,
):
    """
    Complete end-to-end Priority-Guided Region Growing (PGRG) pipeline.

    Execution Order (identical to verified experiment pipeline):
      1. Preprocess or normalize input image.
      2. Compute priority features (IDV, GM, TV).
      3. Compute priority map (weighted linear combination, equal 1/3 each by default).
      4. Priority-guided region growing using max-heap (8-connected).
      5. Construct Disjoint-Set Union (Union-Find) on accepted pixels.
      6. Filter regions by minimum area (removes components with area < min_area).
      7. Morphological closing (bridges fragmented segments if merge_gap > 0).

    Args:
        image (str or numpy.ndarray): Image file path or 2D/3D numpy array.
        seed_threshold (float): Priority threshold for seed selection (default: 0.45).
        similarity_threshold (float): Intensity difference threshold for growing (default: 25).
        min_area (int): Minimum connected component area to retain (default: 100).
        merge_gap (int): Morphological closing structuring element size in pixels (default: 11).
                         If 0, no morphological closing is applied.
        weights (dict or tuple, optional): Feature weights for priority map (default: None, equal).
        return_intermediate (bool): If True, returns (filtered_mask, intermediate_dict).

    Returns:
        numpy.ndarray: Binary defect mask (H, W) uint8 with values in {0, 255}.
    """
    if isinstance(image, str):
        from src.preprocessing import preprocess_image
        _, _, _, norm = preprocess_image(image)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            norm = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            norm = image
    else:
        raise ValueError("image must be an image file path or a numpy.ndarray.")

    # 1. Feature extraction
    _, idv, gm, tv = compute_priority_features(norm)

    # 2. Priority map
    pmap = compute_priority_map(idv, gm, tv, weights=weights)

    # 3. Priority-guided region growing
    rmask, visited, processed_pixels = priority_guided_region_growing(
        norm, pmap,
        seed_threshold=seed_threshold,
        similarity_threshold=similarity_threshold
    )

    # 4. Union-Find connected component building
    uf, reg_idx = build_region_uf(rmask)

    # 5. Connected component area filtering
    filtered_mask, region_stats = filter_regions(rmask, uf, reg_idx, min_area=min_area)

    # 6. Morphological closing for region consolidation
    if merge_gap > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_gap, merge_gap))
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

    if return_intermediate:
        return filtered_mask, {
            "normalized": norm,
            "priority_map": pmap,
            "region_mask": rmask,
            "visited": visited,
            "processed_pixels": processed_pixels,
            "region_stats": region_stats,
            "uf": uf,
            "region_indices": reg_idx,
        }

    return filtered_mask


def original_pgrg(image, return_intermediate=False):
    """
    Run Original PGRG:
      seed_threshold=0.45, similarity_threshold=25, min_area=100, merge_gap=11, equal weights.
    """
    return pgrg_pipeline(image, **ORIGINAL_CONFIG, return_intermediate=return_intermediate)


def improved_pgrg(image, return_intermediate=False):
    """Alias to original_pgrg."""
    return original_pgrg(image, return_intermediate=return_intermediate)


