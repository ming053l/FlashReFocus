#!/usr/bin/env python
"""Blur-matched focus sweep: for each scene, find the FlashReFocus strength whose out-of-focus blur matches GenRefocus'
(mean absolute Laplacian over pixels far from every focal depth, relative to the AiF input, averaged over the eight focal
settings), re-render the eight FlashReFocus frames at that strength into <scene>/ours_matched/, and record it.
Usage: RACK_OUT=... RACK_AIF=... rack_match_blur.py"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

import fast_refocus as FR
import rack_analyze as RA

R = Path(os.environ.get("RACK_OUT", "/ssd1/chingheng/Defocus/refocus_bench/rack_focus"))
AIF = Path(os.environ.get("RACK_AIF", "/ssd1/chingheng/Defocus/refocus_bench/3creal_lvcorr/aif"))


def sharp(img, m):
    g = img.mean(-1)
    return float(np.abs(ndimage.laplace(g))[m].mean())


def main():
    FR.setup_backend(True)
    rd = FR.FastRenderer()
    meta = json.loads((R / "meta_ours.json").read_text())
    out = {}
    for s in meta["scenes"]:
        o = R / s; pts = json.loads((o / "points.json").read_text()); d = np.load(o / "depth.npy")
        aif = RA.load(AIF / f"{s}.png"); size = (aif.shape[1], aif.shape[0])
        masks = [np.abs(d - p["depth"]) > 0.25 for p in pts]          # clearly out of focus for that setting
        base = [sharp(aif, m) for m in masks]
        gen = [sharp(RA.load(o / "genrefocus" / f"{i:02d}.png", size), m) / b for i, (m, b) in enumerate(zip(masks, base))]
        target = float(np.mean(gen))                                    # remaining sharpness ratio (lower = blurrier)
        rd.set_image(AIF / f"{s}.png")

        def ours_ratio(st):
            r = []
            for i, (p, m, b) in enumerate(zip(pts, masks, base)):
                pr = rd.render(p["x"], p["y"], st, None)[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                r.append(sharp(pr, m) / b)
            return float(np.mean(r))

        lo, hi = 0.0, 1.0                                               # ratio decreases with strength
        r_hi = ours_ratio(hi)
        if r_hi > target:
            st = hi                                                     # cannot blur as much: use the maximum
        else:
            for _ in range(12):
                mid = (lo + hi) / 2
                if ours_ratio(mid) > target:
                    lo = mid
                else:
                    hi = mid
            st = (lo + hi) / 2
        (o / "ours_matched").mkdir(exist_ok=True)
        for i, p in enumerate(pts):
            rd.render(p["x"], p["y"], st, o / "ours_matched" / f"{i:02d}.png")
        out[s] = {"genrefocus_sharpness_ratio": target, "matched_strength": st, "ours_ratio_at_matched": ours_ratio(st),
                  "ours_ratio_at_0.5": ours_ratio(0.5), "ours_ratio_at_1.0": r_hi}
        print(s, out[s], flush=True)
    (R / "blur_match.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
