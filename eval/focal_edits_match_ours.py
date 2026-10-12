#!/usr/bin/env python
"""Focal-edit study, setting (i): FlashReFocus matched to the blur of GenRefocus at K = 15. Blur is measured as the
out-of-focus sharpness ratio: mean |Laplacian| over pixels with |D - D_f| > 0.25, relative to the all-in-focus input.

The strength is searched in [0, 2]: above 1 the renderer extrapolates beyond its trained range and blur is not always
monotonic there, so only up to the blurriest of 1.25 .. 2; if the result is still sharper than GenRefocus, the
renderer is applied twice. Matched per focal setting -> <scene>/ours_matched_per_setting (in-focus fidelity, figures);
one strength per scene matching the blur averaged over the eight settings -> <scene>/ours_matched_per_scene
(consistency). Also records FlashReFocus' ratio at strength 0.5, the target of setting (ii)
(focal_edits_match_genrefocus.py). Writes blur_match_ours.json."""
import argparse
import json

import numpy as np
import torch

from flashrefocus import Refocuser, Scene
from flashrefocus.image_io import save_image
from eval_utils import laplacian_sharpness, load_rgb
from focal_edits_common import AIF, OUT, OUT_OF_FOCUS, scene_dirs

MAX_STRENGTH = 2.0


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    torch.backends.cudnn.benchmark = False   # reproducible: autotuning may pick other algorithms between runs
    rf = Refocuser.from_pretrained(deblurring=False)
    result = {}
    for scene_dir in scene_dirs():
        s = scene_dir.name
        points = json.loads((scene_dir / "points.json").read_text())
        depth = np.load(scene_dir / "depth.npy")
        aif = load_rgb(AIF / f"{s}.png")
        size = (aif.shape[1], aif.shape[0])
        scene = rf.scene_from_file(AIF / f"{s}.png")
        focal_depths = [scene.focal_depth_at(p["x"], p["y"]) for p in points]
        masks = [np.abs(depth - p["depth"]) > OUT_OF_FOCUS for p in points]
        base = [laplacian_sharpness(aif, m) for m in masks]
        target = [laplacian_sharpness(load_rgb(scene_dir / "genrefocus" / f"{i:02d}.png", size), m) / b
                  for i, (m, b) in enumerate(zip(masks, base))]

        def render(i, strength, passes):
            current = scene
            for _ in range(passes):
                current = Scene(rf.render(current, strength, focal_depths[i]).clamp(0, 1), scene.depth)
            return current.image

        def ratio(indices, strength, passes):
            return float(np.mean([laplacian_sharpness(render(i, strength, passes)[0].permute(1, 2, 0).cpu().numpy(),
                                                      masks[i]) / base[i] for i in indices]))

        def bisect(indices, goal, lo, hi, passes):          # the ratio decreases with the strength
            for _ in range(12):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if ratio(indices, mid, passes) > goal else (lo, mid)
            return (lo + hi) / 2

        def search(indices, goal):
            """One pass in [0, 1]; one pass up to the blurriest extrapolated strength s*; two passes in [0, 1], then
            up to s*; otherwise the blurriest setting tried."""
            if ratio(indices, 1.0, 1) <= goal:
                return bisect(indices, goal, 0.0, 1.0, 1), 1
            s_star = min([1.25, 1.5, 1.75, MAX_STRENGTH], key=lambda k: ratio(indices, k, 1))
            if ratio(indices, s_star, 1) <= goal:
                return bisect(indices, goal, 1.0, s_star, 1), 1
            if ratio(indices, 1.0, 2) <= goal:
                return bisect(indices, goal, 0.0, 1.0, 2), 2
            if ratio(indices, s_star, 2) <= goal:
                return bisect(indices, goal, 1.0, s_star, 2), 2
            return s_star, 2

        per_setting = []
        for sub in ("ours_matched_per_setting", "ours_matched_per_scene"):
            (scene_dir / sub).mkdir(exist_ok=True)
        for i in range(len(points)):
            k, passes = search([i], target[i])
            per_setting.append({"strength": k, "passes": passes, "target": target[i], "ours": ratio([i], k, passes)})
            save_image(render(i, k, passes), scene_dir / "ours_matched_per_setting" / f"{i:02d}.png")
        all_settings = range(len(points))
        k, passes = search(all_settings, float(np.mean(target)))
        for i in all_settings:
            save_image(render(i, k, passes), scene_dir / "ours_matched_per_scene" / f"{i:02d}.png")
        result[s] = {"per_setting": per_setting,
                     "per_scene": {"strength": k, "passes": passes, "target": float(np.mean(target)),
                                   "ours": ratio(all_settings, k, passes)},
                     "ours_ratio_at_0.5": ratio(all_settings, 0.5, 1)}
        print(s, [(round(x["strength"], 2), x["passes"]) for x in per_setting], "scene:", (round(k, 2), passes),
              flush=True)
    (OUT / "blur_match_ours.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
