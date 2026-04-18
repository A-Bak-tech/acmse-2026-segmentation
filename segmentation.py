"""
ACMSE 2026 Undergraduate Programming Contest
AI Image Segmentation (Grayscale)

Algorithm : Union-Find (Disjoint Set Union) with path-halving compression
            and union by rank.

Segmentation rule
-----------------
Two pixels belong to the same segment if and only if:
  1. They are 4-directionally adjacent (up / down / left / right).
  2. Their absolute grayscale intensity difference is <= 15.

Allowed libraries: NumPy, Pillow — no external segmentation libraries used.
"""

import numpy as np
from PIL import Image
import os


# ─────────────────────────────────────────────────────────────────────────────
#  Core segmentation
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD = 15          # maximum allowed intensity difference between neighbours


def segment_image(img_array: np.ndarray) -> np.ndarray:
    """
    Assign a dense segment ID to every pixel of a grayscale image.

    Parameters
    ----------
    img_array : ndarray, shape (H, W), dtype uint8
        Grayscale pixel intensities.

    Returns
    -------
    ndarray, shape (H, W), dtype int32
        Segment IDs starting at 0, densely packed (no gaps).
    """
    H, W = img_array.shape
    N    = H * W
    flat = img_array.astype(np.int32).ravel()   # 1-D view for fast lookup

    # --- Union-Find arrays ---------------------------------------------------
    parent = np.arange(N, dtype=np.int32)       # each pixel starts as its own root
    rank   = np.zeros(N, dtype=np.int32)

    def find(x: int) -> int:
        """Iterative find with path-halving (cuts path length in half)."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]       # point to grandparent
            x = parent[x]
        return x

    def unite(a: int, b: int) -> None:
        """Union by rank."""
        a, b = find(a), find(b)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    # --- Build pixel index grid ----------------------------------------------
    idx = np.arange(N, dtype=np.int32).reshape(H, W)

    # --- Horizontal edges: pixel (r,c) — (r, c+1) ----------------------------
    left  = idx[:, :-1].ravel()
    right = idx[:, 1: ].ravel()
    h_keep = np.abs(flat[left] - flat[right]) <= THRESHOLD
    for a, b in zip(left[h_keep].tolist(), right[h_keep].tolist()):
        unite(a, b)

    # --- Vertical edges: pixel (r,c) — (r+1, c) ------------------------------
    top = idx[:-1, :].ravel()
    bot = idx[1:,  :].ravel()
    v_keep = np.abs(flat[top] - flat[bot]) <= THRESHOLD
    for a, b in zip(top[v_keep].tolist(), bot[v_keep].tolist()):
        unite(a, b)

    # --- Assign dense labels -------------------------------------------------
    roots = np.array([find(i) for i in range(N)], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.reshape(H, W).astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_grayscale(path: str) -> np.ndarray:
    """Load any image as a 2-D uint8 grayscale array."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def save_mask(mask: np.ndarray, out_path: str) -> None:
    """
    Save a segment mask as a 16-bit PNG.

    Pixel values are raw segment IDs.  The image will look nearly black in
    standard viewers — this is expected.  Use numpy.unique() to inspect IDs
    or call save_mask_visual() for a normalised preview.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    Image.fromarray(mask.astype(np.uint16)).save(out_path)


def save_mask_visual(mask: np.ndarray, out_path: str) -> None:
    """Save a contrast-stretched version (0-255) for visual inspection only."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    top = mask.max()
    vis = ((mask.astype(np.float32) / max(top, 1)) * 255).astype(np.uint8)
    Image.fromarray(vis).save(out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test = np.array([
        [10, 12, 80, 82],
        [11, 14, 79, 83],
        [50, 52, 30, 31],
        [51, 53, 28, 29],
    ], dtype=np.uint8)

    seg = segment_image(test)
    print("Segment map:\n", seg)
    print("Unique IDs:", np.unique(seg))
    assert seg[0, 0] == seg[1, 1],  "Top-left 2×2 block should be one segment"
    assert seg[0, 0] != seg[0, 2],  "Top-left and top-right should differ"
    assert seg[2, 0] != seg[0, 0],  "Mid-left block should differ from top-left"
    print("Smoke test passed ✓")