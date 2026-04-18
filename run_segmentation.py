import argparse
import os
import time

import numpy as np
from PIL import Image

from segmentation import load_grayscale, save_mask, save_mask_visual, segment_image


def boundary_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    def bmap(seg):
        b = np.zeros(seg.shape, dtype=bool)
        b[:, :-1] |= seg[:, :-1] != seg[:, 1:]
        b[:, 1:]  |= seg[:, :-1] != seg[:, 1:]
        b[:-1, :] |= seg[:-1, :] != seg[1:, :]
        b[1:,  :] |= seg[:-1, :] != seg[1:, :]
        return b
    return float((bmap(pred) == bmap(gt)).mean())


def run_test(input_dir: str, output_dir: str, visual: bool = False):
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".png"))
    print(f"[test] {len(files)} images  →  {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = output_dir + "_visual" if visual else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    timings = []
    for fname in files:
        img = load_grayscale(os.path.join(input_dir, fname))
        t0 = time.perf_counter()
        seg = segment_image(img)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        save_mask(seg, os.path.join(output_dir, fname))
        if vis_dir:
            save_mask_visual(seg, os.path.join(vis_dir, fname))
        print(f"  {fname:20s}  {img.shape}  segs={seg.max()+1:6d}  {elapsed:.2f}s")

    print(f"\nAvg {np.mean(timings):.2f}s/image  |  Total {sum(timings):.1f}s")


def run_eval(input_dir: str, mask_dir: str, output_dir: str):
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".png"))
    print(f"[eval] {len(files)} images")
    os.makedirs(output_dir, exist_ok=True)

    scores = []
    for fname in files:
        gt_path = os.path.join(mask_dir, fname)
        if not os.path.exists(gt_path):
            print(f"  [skip] {fname} — no ground-truth found")
            continue
        img = load_grayscale(os.path.join(input_dir, fname))
        seg = segment_image(img)
        save_mask(seg, os.path.join(output_dir, fname))
        gt = np.array(Image.open(gt_path)).astype(np.int32)
        acc = boundary_accuracy(seg, gt)
        scores.append(acc)
        print(f"  {fname:20s}  boundary_acc={acc:.4f}  pred_segs={seg.max()+1}")

    if scores:
        print(f"\nMean boundary accuracy: {np.mean(scores):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",   choices=["test", "eval"], default="test")
    p.add_argument("--input",  default="contest_dataset/test/images")
    p.add_argument("--output", default="predictions")
    p.add_argument("--masks",  default="contest_dataset/train/masks")
    p.add_argument("--visual", action="store_true")
    args = p.parse_args()

    if args.mode == "test":
        run_test(args.input, args.output, visual=args.visual)
    else:
        run_eval(args.input, args.masks, args.output)