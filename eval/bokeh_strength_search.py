#!/usr/bin/env python
"""FlashReFocus under the bokeh-table protocol (Tables 3, G.1): each all-in-focus input is rendered at the annotated
focal point with strengths k/100, k = 1..100, and the output with the highest SSIM to the target is kept (integer
binary search on the unimodal SSIM, as the baselines' K searches). Inputs, targets and focal points in the 3CReal
layout (original/, bokeh/, focus_point.json). Outputs: <out>/images/<key>.png and <out>/results.csv."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from flashrefocus import Refocuser
from flashrefocus.image_io import read_image, save_image
from flashrefocus.pipeline import quantize_to_8_bits


def search_strength(refocuser, scene, focal_depth, target, depth=None, k_min=1, k_max=100):
    """Strength k/100 with the highest SSIM to the target, by integer binary search on the unimodal SSIM.
    Returns (k, SSIM, rendered image quantised to 8 bits, as saved)."""
    cache = {}

    def score(k):
        if k not in cache:
            bokeh = quantize_to_8_bits(refocuser.render(scene, k / 100.0, focal_depth, depth))
            cache[k] = (float(ssim(bokeh, target, data_range=1.0)), bokeh)
        return cache[k][0]

    lo, hi = k_min, k_max
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if score(mid) < score(mid + 1) else (lo, mid)
    score(lo)
    return lo, cache[lo][0], cache[lo][1], len(cache)


def annotated_focal_point(annotation, height, width):
    return annotation["x"] * width / annotation["width"], annotation["y"] * height / annotation["height"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="folder with original/, bokeh/, focus_point.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k_min", type=int, default=1)
    ap.add_argument("--k_max", type=int, default=100)
    a = ap.parse_args()
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False   # reproducible: autotuning may pick other algorithms between runs
    rf = Refocuser.from_pretrained(deblurring=False)
    annotations = json.loads((a.root / "focus_point.json").read_text())["annotations"]
    rows = []
    for key in sorted(annotations, key=int):
        src = a.root / "original" / f"{key}.jpg"
        if not src.exists():
            continue
        scene = rf.scene_from_file(src)
        focal_depth = scene.focal_depth_at(*annotated_focal_point(annotations[key], *scene.image.shape[-2:]))
        target = read_image(a.root / "bokeh" / f"{key}.jpg", "cuda")
        k, score, bokeh, n_evals = search_strength(rf, scene, focal_depth, target, k_min=a.k_min, k_max=a.k_max)
        save_image(bokeh, a.out / "images" / f"{key}.png")
        rows.append({"key": key, "best_k": k, "strength": k / 100.0, "ssim": score, "n_evals": n_evals})
        print(key, k, round(score, 4), flush=True)
    with open(a.out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("mean ssim", sum(r["ssim"] for r in rows) / len(rows))


if __name__ == "__main__":
    main()
