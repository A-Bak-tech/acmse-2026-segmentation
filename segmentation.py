import numpy as np
from PIL import Image
import os

THRESHOLD = 15


def segment_image(img_array: np.ndarray) -> np.ndarray:
    H, W = img_array.shape
    N = H * W
    flat = img_array.astype(np.int32).ravel()
    parent = np.arange(N, dtype=np.int32)
    rank = np.zeros(N, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a, b):
        a, b = find(a), find(b)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    idx = np.arange(N, dtype=np.int32).reshape(H, W)

    left = idx[:, :-1].ravel()
    right = idx[:, 1:].ravel()
    for a, b in zip(left[np.abs(flat[left] - flat[right]) <= THRESHOLD].tolist(),
                    right[np.abs(flat[left] - flat[right]) <= THRESHOLD].tolist()):
        unite(a, b)

    top = idx[:-1, :].ravel()
    bot = idx[1:, :].ravel()
    for a, b in zip(top[np.abs(flat[top] - flat[bot]) <= THRESHOLD].tolist(),
                    bot[np.abs(flat[top] - flat[bot]) <= THRESHOLD].tolist()):
        unite(a, b)

    roots = np.array([find(i) for i in range(N)], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.reshape(H, W).astype(np.int32)


def load_grayscale(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def save_mask(mask: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    Image.fromarray(mask.astype(np.uint16)).save(out_path)


def save_mask_visual(mask: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    vis = ((mask.astype(np.float32) / max(mask.max(), 1)) * 255).astype(np.uint8)
    Image.fromarray(vis).save(out_path)