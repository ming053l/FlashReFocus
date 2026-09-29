#!/usr/bin/env python
"""Sharpness of restored images relative to the ground truth (the paper's Laplacian-variance measure).

For every pair (outputs named {index:04d}.png as in the PASD Table-1 runs) reports the ratio
LapVar(output) / LapVar(GT) of the grey image, at full resolution and after 0.5x bicubic downscaling
(which suppresses pixel-level noise and compares edges), plus PSNR at 0.5x.
A ratio of 1 means as much high-frequency energy as the ground truth; blurry outputs are < 1.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PASD_RUNS = Path("/ssd1/chingheng/Defocus/MyAutoRefocus/runs")
PAIRS = {
    "dpdd": PASD_RUNS / "dpdd_ans_sweep/n400/two_stage_dpdd_cfg1.0_ans400_tile320/validation/step_000000/metrics.json",
    "realdof": PASD_RUNS / "realdof_ans_sweep/n400/two_stage_realdof_cfg1.0_ans400_tile320/validation/step_000000/metrics.json",
}


def lapvar(rgb):
    return cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(PAIRS))
    ap.add_argument("--gt_dir", type=Path, help="instead of --dataset: pair outputs with ground truths of the same name")
    ap.add_argument("--pred_dir", type=Path, required=True)
    ap.add_argument("--pred_pattern", default="{index:04d}.png")
    a = ap.parse_args()
    if a.gt_dir:
        pairs = [(p, a.gt_dir / p.name) for p in sorted(a.pred_dir.glob("*.png"))]
    else:
        pairs = [(a.pred_dir / a.pred_pattern.format(index=s["index"]), Path(s["target_path"]))
                 for s in json.load(open(PAIRS[a.dataset]))["samples"]]
    r_full, r_half, psnr_half = [], [], []
    for p, t in pairs:
        if not p.exists():
            continue
        pred = Image.open(p).convert("RGB")
        gt = Image.open(t).convert("RGB").crop((0, 0) + pred.size)
        half = (pred.size[0] // 2, pred.size[1] // 2)
        P, G = np.asarray(pred), np.asarray(gt)
        Ph, Gh = np.asarray(pred.resize(half, Image.BICUBIC)), np.asarray(gt.resize(half, Image.BICUBIC))
        r_full.append(lapvar(P) / lapvar(G))
        r_half.append(lapvar(Ph) / lapvar(Gh))
        psnr_half.append(10 * np.log10(255.0 ** 2 / np.mean((Ph.astype(np.float64) - Gh) ** 2)))
    print(f"{a.dataset or a.gt_dir.name} n={len(r_full)} sharpness_full {np.mean(r_full):.3f} sharpness_0.5x {np.mean(r_half):.3f} "
          f"PSNR_0.5x {np.mean(psnr_half):.2f}")


if __name__ == "__main__":
    main()
