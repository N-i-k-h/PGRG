"""
union_find.py -- Union-Find (Disjoint Set Union) for Region Merging

Phase 6 implementation:
  - UnionFind        : Reusable DSU class with path compression and union by rank.
  - pixel_index()    : Convert 2D pixel coordinates (y, x) to a flat 1D index.
  - build_region_uf(): Construct a UnionFind over all accepted PGRG region pixels
                       and union every pair of adjacent region pixels.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helper: coordinate mapping
# ---------------------------------------------------------------------------

def pixel_index(y, x, width):
    """
    Convert a 2-D pixel coordinate to a flat 1-D index.

    The mapping follows row-major (C) order:
        index = y * width + x

    Args:
        y (int): Row index (0-based).
        x (int): Column index (0-based).
        width (int): Number of columns in the image.

    Returns:
        int: Flat 1-D index for the pixel.
    """
    return y * width + x


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure.

    Supports efficient connectivity queries with:
      - Path compression in find()
      - Union by rank in union()

    Both operations run in near O(1) amortised time (inverse Ackermann).

    Args:
        size (int): Total number of elements (pixels) to track.
                    Elements are labelled 0 ... size-1.
    """

    def __init__(self, size):
        """
        Initialise the Union-Find structure.

        Args:
            size (int): Number of elements.

        Attributes:
            parent (list[int]): parent[i] == i initially (each element is its
                                own root).
            rank   (list[int]): rank[i] == 0 initially; used to keep trees
                                shallow during union operations.
            n_size (int)      : Total number of elements.
        """
        if size < 0:
            raise ValueError("size must be non-negative, got {}".format(size))

        self.n_size = size
        # Each element starts as its own representative
        self.parent = list(range(size))
        # Rank (upper-bound on tree height) starts at 0
        self.rank = [0] * size

    # ------------------------------------------------------------------
    # find -- with full iterative path compression
    # ------------------------------------------------------------------

    def find(self, x):
        """
        Find the representative (root) of the set containing element x.

        Applies path compression: every node along the path to the root
        is re-parented directly to the root, flattening the tree for all
        future queries.

        Args:
            x (int): Element to look up (0-based index).

        Returns:
            int: Root representative of the set containing x.

        Raises:
            IndexError: If x is out of range [0, n_size).
        """
        if not (0 <= x < self.n_size):
            raise IndexError(
                "Element {} is out of range [0, {})".format(x, self.n_size)
            )

        # Iterative path compression -- avoids Python recursion depth limit
        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        # Second pass: redirect every node on the path directly to root
        while self.parent[x] != root:
            next_node = self.parent[x]
            self.parent[x] = root
            x = next_node

        return root

    # ------------------------------------------------------------------
    # union -- with union by rank
    # ------------------------------------------------------------------

    def union(self, x, y):
        """
        Merge the sets containing elements x and y.

        Uses union by rank: attach the shorter tree under the taller one.
        When ranks are equal, root of x's set becomes the new root and its
        rank is incremented by one.

        Args:
            x (int): First element.
            y (int): Second element.

        Returns:
            bool: True if a merge occurred (different sets);
                  False if already in the same set.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            # Already in the same component -- no merge needed
            return False

        # Attach shorter tree under taller tree to bound tree depth
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            # Equal rank: root_x absorbs root_y, increment rank
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    # ------------------------------------------------------------------
    # connected -- membership query
    # ------------------------------------------------------------------

    def connected(self, x, y):
        """
        Return True if elements x and y belong to the same set.

        Args:
            x (int): First element.
            y (int): Second element.

        Returns:
            bool: True if find(x) == find(y).
        """
        return self.find(x) == self.find(y)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def count_roots(self):
        """
        Count the number of distinct sets (components) across ALL elements.

        Returns:
            int: Number of unique root representatives.
        """
        return sum(1 for i in range(self.n_size) if self.parent[i] == i)

    def unique_roots_for(self, indices):
        """
        Return the set of unique roots for a subset of element indices.

        Args:
            indices (iterable of int): Element indices to examine.

        Returns:
            set: Unique root representatives for the given indices.
        """
        return {self.find(i) for i in indices}


# ---------------------------------------------------------------------------
# Integration helper: build Union-Find over PGRG region pixels
# ---------------------------------------------------------------------------

# 8-neighbourhood offsets (identical to those used in Phase 5 PGRG)
_NEIGHBORS_8 = [
    (-1, -1), (-1,  0), (-1,  1),
    ( 0, -1),           ( 0,  1),
    ( 1, -1), ( 1,  0), ( 1,  1),
]


def build_region_uf(region_mask):
    """
    Construct a UnionFind covering all pixels in region_mask and union
    every pair of adjacent (8-connected) accepted pixels.

    The flat index for pixel (y, x) in an image of width w is:
        index = y * w + x

    Only pixels where region_mask[y, x] == 1 are considered.
    Adjacent accepted pixel pairs are union-ed so the structure tracks
    which region pixels are mutually reachable (connected components).

    This function does NOT modify region_mask or any Phase 1-5 data.

    Args:
        region_mask (numpy.ndarray): Binary uint8 mask (H, W) with 1 for
                                     accepted defect pixels, 0 elsewhere.

    Returns:
        tuple: (uf, region_indices)
            - uf (UnionFind)        : Fully constructed Union-Find for the
                                      image (size = H * W).
            - region_indices (list[int]): Flat 1-D indices of all pixels
                                          that belong to the PGRG region.
    """
    h, w = region_mask.shape
    total_pixels = h * w

    # One DSU element per pixel in the full image
    uf = UnionFind(total_pixels)

    # Coordinates of all accepted region pixels
    ys, xs = np.where(region_mask == 1)

    # Flat index list for accepted pixels
    region_indices = [pixel_index(int(y), int(x), w) for y, x in zip(ys, xs)]

    # Union each accepted pixel with all its accepted 8-neighbours
    for y, x in zip(ys, xs):
        idx = pixel_index(int(y), int(x), w)
        for dy, dx in _NEIGHBORS_8:
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < h and 0 <= nx < w and region_mask[ny, nx] == 1:
                n_idx = pixel_index(ny, nx, w)
                uf.union(idx, n_idx)

    return uf, region_indices
