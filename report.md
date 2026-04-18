# ACMSE 2026 Undergraduate Programming Contest
## Grayscale Image Segmentation

---

## 1. Problem Overview

Segment each grayscale image into contiguous regions where two pixels belong to the same segment if and only if:

1. They are **4-directionally adjacent** (up, down, left, right).
2. Their absolute grayscale intensity difference is **≤ 15**.

---

## 2. Algorithm: Union-Find

This is a connected-components problem solved with **Union-Find (Disjoint Set Union)**.

### Steps

1. **Initialize** `parent[i] = i` and `rank[i] = 0` for all N pixels.
2. **Extract edges** using NumPy slicing — all horizontal pairs `(r,c)—(r,c+1)` and vertical pairs `(r,c)—(r+1,c)` where the intensity difference is ≤ 15.
3. **Union** each valid pair using union by rank.
4. **Find** the root of every pixel using path-halving compression.
5. **Relabel** roots to dense consecutive IDs starting from 0.

### Complexity

O(H·W) time and space. Path-halving gives near-constant amortized find operations.

---

## 3. Implementation

- **Language**: Python 3
- **Libraries**: NumPy, Pillow
- No external segmentation libraries used.

Masks are saved as 16-bit PNGs to support images with more than 255 segments. Values are raw segment IDs and will appear nearly black in standard image viewers — use `numpy.unique()` to inspect.

---

## 4. Results

Evaluated on 100 training images (321×481 or 481×321 pixels):

| Metric | Value |
|--------|-------|
| Mean boundary accuracy | 0.8128 |
| Average time per image | ~0.28s |
| Segment count range | 338 – 37,490 |

---

## 5. Usage

```bash
# Generate test masks
python run_segmentation.py --mode test \
    --input contest_dataset/test/images \
    --output predictions

# Evaluate on training set
python run_segmentation.py --mode eval \
    --input contest_dataset/train/images \
    --masks contest_dataset/train/masks \
    --output train_predictions
```

---

## 6. File Structure

```
├── segmentation.py       # Core Union-Find segmentation algorithm
├── run_segmentation.py   # Batch runner
├── predictions/          # Output masks for test set (50 PNGs)
└── report.md
```