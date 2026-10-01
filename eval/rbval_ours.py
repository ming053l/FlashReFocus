#!/usr/bin/env python
"""FlashReFocus Stage II under the bokeh-table protocol: each all-in-focus input is rendered at the annotated focal point with
strengths k/100, k = 1..100, and the output with the highest SSIM to the target is kept (integer binary search on the
unimodal SSIM, as the baselines' K searches). Inputs, targets and focus points in the 3CReal layout
(original/, bokeh/, focus_point.json). Outputs: <out>/images/<key>.png and <out>/results.csv."""
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

import fast_refocus as FR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="folder with original/, bokeh/, focus_point.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k_min", type=int, default=1)
    ap.add_argument("--k_max", type=int, default=100)
    a = ap.parse_args()
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    FR.setup_backend(True)
    rd = FR.FastRenderer()
    ann = json.loads((a.root / "focus_point.json").read_text())["annotations"]
    rows = []
    for key in sorted(ann, key=lambda k: int(k)):
        src = a.root / "original" / f"{key}.jpg"
        if not src.exists():
            continue
        rd.set_image(src)
        h, w = rd.src.shape[-2:]
        fx = ann[key]["x"] * w / ann[key]["width"]; fy = ann[key]["y"] * h / ann[key]["height"]
        gt = FR.read_rgb(a.root / "bokeh" / f"{key}.jpg", "cuda")
        cache = {}

        def score(k):
            if k not in cache:
                with torch.no_grad():
                    p = rd.render(fx, fy, k / 100.0, None).float().clamp(0, 1)
                    # the saved PNG is quantised: score the quantised result
                    p = (p * 255.0).to(torch.uint8).float() / 255.0
                    cache[k] = (float(ssim(p, gt, data_range=1.0)), p)
            return cache[k][0]

        lo, hi = a.k_min, a.k_max
        while lo < hi:
            mid = (lo + hi) // 2
            if score(mid) < score(mid + 1):
                lo = mid + 1
            else:
                hi = mid
        best = lo; score(best)
        FR.write_rgb(cache[best][1], a.out / "images" / f"{key}.png")
        rows.append({"key": key, "best_k": best, "strength": best / 100.0, "ssim": cache[best][0], "n_evals": len(cache)})
        print(key, best, round(cache[best][0], 4), flush=True)
    with open(a.out / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print("mean ssim", sum(r["ssim"] for r in rows) / len(rows))


if __name__ == "__main__":
    main()
