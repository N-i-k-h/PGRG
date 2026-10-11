"""
region_filtering.py -- Region Filtering and Final Binary Defect Mask

Phase 7 implementation:
  - compute_region_stats(): Compute area of each connected component from UnionFind.
  - filter_regions()      : Remove small regions (area < min_area) and produce the
                            final binary defect mask (0 = background, 255 = defect).

This module consumes the outputs of Phase 5 (region_mask) and Phase 6
(UnionFind + region_indices) without modifying them.
"""

import numpy as np

from src.union_find import pixel_index


def compute_region_stats(uf, region_indices, width):
    """
    Compute area (pixel count) for each connected region identified by the
    Union-Find structure.

    Args:
        uf (UnionFind)          : Fully-constructed Union-Find from Phase 6.
        region_indices (list[int]): Flat 1-D indices of all PGRG region pixels.
        width (int)             : Image width (columns), used for index mapping.

    Returns:
        dict: Mapping from root representative (int) to area (int).
              Only roots that own at least one region pixel are included.
    """
    area_by_root = {}
    for idx in region_indices:
        root = uf.find(idx)
        area_by_root[root] = area_by_root.get(root, 0) + 1
    return area_by_root


def filter_regions(region_mask, uf, region_indices, min_area=20):
    """
    Filter connected regions by area and produce the final binary defect mask.

    Algorithm:
      1. Use the Union-Find to compute the area of every connected component
         among the accepted PGRG pixels.
      2. Remove (zero out) regions whose area < min_area.
      3. Keep regions whose area >= min_area.
      4. Produce the final mask: background = 0, defect = 255.

    This function does NOT modify region_mask or the Union-Find.

    Args:
        region_mask (numpy.ndarray): Binary uint8 mask (H, W) from Phase 5,
                                     with 1 for accepted pixels.
        uf (UnionFind)             : Union-Find from Phase 6.
        region_indices (list[int]) : Flat 1-D indices of region pixels.
        min_area (int)             : Minimum number of pixels a connected
                                     region must contain to be retained.
                                     This is configurable; do NOT hard-code
                                     a final experimental value.

    Returns:
        tuple: (filtered_mask, region_statistics)
            - filtered_mask (numpy.ndarray) : Binary uint8 mask (H, W) with
                                              0 = background, 255 = defect.
            - region_statistics (list[dict]): One entry per connected region
                                              (before filtering), each with:
                  * 'root'   : int   -- Union-Find root representative.
                  * 'area'   : int   -- Number of pixels in the region.
                  * 'status' : str   -- 'KEPT' or 'REMOVED'.
    """
    h, w = region_mask.shape

    # 1. Compute area per connected component
    area_by_root = compute_region_stats(uf, region_indices, w)

    # 2. Classify each region as KEPT or REMOVED
    region_statistics = []
    kept_roots = set()
    for root, area in sorted(area_by_root.items()):
        if area >= min_area:
            status = "KEPT"
            kept_roots.add(root)
        else:
            status = "REMOVED"
        region_statistics.append({
            "root": root,
            "area": area,
            "status": status,
        })

    # 3. Build filtered mask: iterate over region pixels, keep only those
    #    belonging to a KEPT root; output 255 for defect, 0 for background.
    filtered_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in region_indices:
        root = uf.find(idx)
        if root in kept_roots:
            y = idx // w
            x = idx % w
            filtered_mask[y, x] = 255

    return filtered_mask, region_statistics
