# ACMSE 2026 Undergraduate Programming Contest
## Report: Grayscale Image Segmentation via Union-Find

---

## 1. Problem Overview

The task is to segment each grayscale image into contiguous regions (segments) based on local pixel similarity. Two pixels belong to the same segment if and only if:

1. They are **4-directionally adjacent** (up, down, left, right — no diagonals).
2. Their absolute grayscale intensity difference is **≤ 15**.

Every pixel must be assigned to exactly one segment. The output for each image is a mask PNG where each pixel value is the integer segment ID.

---

## 2. Algorithm: Union-Find (Disjoint Set Union)

The segmentation rule defines a connected-components problem: find all maximal connected subgraphs of the pixel grid where each edge satisfies the ≤ 15 intensity threshold. We solve this with the **Union-Find (DSU)** data structure, which efficiently tracks and merges component membership.

### 2.1 Data Structures

- **`parent[i]`** — stores the representative (root) of pixel `i`'s component. Initialized to `parent[i] = i` (each pixel is its own component).
- **`rank[i]`** — tree depth heuristic for balanced merging.

### 2.2 Algorithm Steps

1. **Initialization**: Allocate `parent` and `rank` arrays of length `H × W`, indexing pixels in row-major order.

2. **Find edges**: Using NumPy vectorized operations, extract all horizontal pairs `(r, c) — (r, c+1)` and all vertical pairs `(r, c) — (r+1, c)`. For each pair, compute the absolute intensity difference. Retain only pairs where this difference is ≤ 15.

3. **Union**: For each retained edge `(a, b)`, call `union(a, b)`. The `find` operation uses **path-halving** (pointing each node to its grandparent during traversal), which achieves near-constant amortized time. The `union` operation uses **union by rank** to keep trees shallow.

4. **Label extraction**: After all unions, call `find(i)` for every pixel `i` to resolve final roots. Map the unique roots to dense consecutive IDs (0, 1, 2, …) using `numpy.unique`.

### 2.3 Complexity

| Step | Time | Space |
|------|------|-------|
| Edge extraction (NumPy) | O(H·W) | O(H·W) |
| Union-Find operations | O(H·W · α(H·W)) ≈ O(H·W) | O(H·W) |
| Label compaction | O(H·W log H·W) | O(H·W) |

Where α is the inverse Ackermann function, effectively constant. Total complexity is **O(H·W)** in practice.

---

## 3. Implementation Details

- **Language**: Python 3
- **Libraries**: NumPy (array operations), Pillow (image I/O)
- No external segmentation libraries are used.

**Edge vectorization**: Rather than iterating over every pixel in Python, NumPy array slicing produces all horizontal and vertical index pairs in a single operation. Only pairs satisfying the threshold are passed to the Python union loop, minimizing iterations.

**Output format**: Masks are saved as 16-bit grayscale PNGs (`mode='I'` in Pillow) to support images with more than 255 segments. Raw segment ID values are preserved exactly — the images appear nearly black in standard viewers, which is expected.

**Path-halving**: During `find(x)`, each visited node is redirected to its grandparent (`parent[x] = parent[parent[x]]`), flattening the tree incrementally without a second pass.

---

## 4. Results

Tested on 96 grayscale images (321×481 or 481×321 pixels):

| Metric | Value |
|--------|-------|
| Average processing time | ~0.51 s/image |
| Segments per image (range) | 338 – 37,490 |
| Segments per image (typical) | 2,000 – 10,000 |

The wide range in segment count reflects image complexity: smooth sky/water images produce few segments, while high-texture images (foliage, rock faces, fur) produce many thousands.

---

## 5. Running the Code

```bash
# Generate masks for test images
python run_segmentation.py --mode test \
    --input  contest_dataset/test/images \
    --output predictions

# Evaluate on training set (requires ground-truth masks)
python run_segmentation.py --mode eval \
    --input  contest_dataset/train/images \
    --masks  contest_dataset/train/masks \
    --output train_predictions

# Smoke test the core algorithm
python segmentation.py
```

---

## 6. File Structure

```
├── segmentation.py        # Core algorithm (Union-Find segmentation)
├── run_segmentation.py    # Batch runner (test + eval modes)
├── predictions/           # Output masks for test set (50 PNGs)
└── report.md              # This document
```