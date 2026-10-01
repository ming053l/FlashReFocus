#!/usr/bin/env python
"""Blur matching with an extended strength range: FlashReFocus strength is searched in [0, 2] (above 1 extrapolates beyond
the trained range, and blur is not always monotonic there, so it is searched only up to the blurriest of 1.25..2); if that
is still sharper than GenRefocus, the renderer is applied twice (strength in [0, 1], then up to that value). Matching per focal setting -> <scene>/ours_matched_pf_ext/ (figures, in-focus fidelity); one setting per scene
matching the blur averaged over the eight settings -> <scene>/ours_matched_ext/ (edit consistency).
Sharpness ratio as in rack_match_blur.py. Usage: RACK_OUT=... RACK_AIF=... rack_match_blur_ext.py"""
import json
import os
from pathlib import Path

import numpy as np

import fast_refocus as FR
import rack_analyze as RA
from rack_match_blur import sharp

R = Path(os.environ.get("RACK_OUT", "/ssd1/chingheng/Defocus/refocus_bench/rack_focus"))
AIF = Path(os.environ.get("RACK_AIF", "/ssd1/chingheng/Defocus/refocus_bench/3creal_lvcorr/aif"))
SMAX = 2.0


def main():
    FR.setup_backend(True)
    rd = FR.FastRenderer()
    meta = json.loads((R / "meta_ours.json").read_text())
    out = {}
    for s in meta["scenes"]:
        o = R / s; pts = json.loads((o / "points.json").read_text()); d = np.load(o / "depth.npy")
        aif = RA.load(AIF / f"{s}.png"); size = (aif.shape[1], aif.shape[0])
        rd.set_image(AIF / f"{s}.png"); src0 = rd.src.clone()
        masks = [np.abs(d - p["depth"]) > 0.25 for p in pts]; base = [sharp(aif, m) for m in masks]
        tgt = [sharp(RA.load(o / "genrefocus" / f"{i:02d}.png", size), m) / b for i, (m, b) in enumerate(zip(masks, base))]

        def render(i, st, passes):
            rd.src = src0.clone()
            for _ in range(passes):
                y = rd.render(pts[i]["x"], pts[i]["y"], st, None).float().clamp(0, 1); rd.src = y
            rd.src = src0.clone()
            return y

        def ratio(idx, st, passes):
            return float(np.mean([sharp(render(i, st, passes)[0].permute(1, 2, 0).cpu().numpy(), masks[i]) / base[i] for i in idx]))

        def bisect(idx, target, lo, hi, passes):
            for _ in range(12):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if ratio(idx, mid, passes) > target else (lo, mid)
            return (lo + hi) / 2

        def search(idx, target):
            # blur is monotonic in [0, 1] but not always beyond it, so extend step by step:
            # one pass in [0, 1]; one pass up to the blurriest extrapolated strength s*; two passes in [0, 1], then up to s*
            if ratio(idx, 1.0, 1) <= target:
                return bisect(idx, target, 0.0, 1.0, 1), 1
            grid = [1.25, 1.5, 1.75, SMAX]
            s_star = min(grid, key=lambda st: ratio(idx, st, 1))
            if ratio(idx, s_star, 1) <= target:
                return bisect(idx, target, 1.0, s_star, 1), 1
            if ratio(idx, 1.0, 2) <= target:
                return bisect(idx, target, 0.0, 1.0, 2), 2
            if ratio(idx, s_star, 2) <= target:
                return bisect(idx, target, 1.0, s_star, 2), 2
            return s_star, 2                                # still sharper: the blurriest setting tried

        per = []
        (o / "ours_matched_pf_ext").mkdir(exist_ok=True); (o / "ours_matched_ext").mkdir(exist_ok=True)
        for i in range(len(pts)):
            st, ps = search([i], tgt[i]); per.append({"strength": st, "passes": ps, "target": tgt[i], "ours": ratio([i], st, ps)})
            FR.write_rgb(render(i, st, ps), o / "ours_matched_pf_ext" / f"{i:02d}.png")
        st, ps = search(range(len(pts)), float(np.mean(tgt)))
        for i in range(len(pts)):
            FR.write_rgb(render(i, st, ps), o / "ours_matched_ext" / f"{i:02d}.png")
        out[s] = {"per_setting": per, "per_scene": {"strength": st, "passes": ps, "target": float(np.mean(tgt)),
                                                     "ours": ratio(range(len(pts)), st, ps)}}
        print(s, [(round(x["strength"], 2), x["passes"]) for x in per], "scene:", (round(st, 2), ps), flush=True)
    (R / "blur_match_ext.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
