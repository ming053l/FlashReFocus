#!/usr/bin/env python
"""Two focal planes by depth editing (Appendix H, Figs. H.9 and H.12): renders plane A only, plane B only and both
(depth_edits.two_focal_planes), measures the sharpness around each plane (mean |Laplacian| relative to the input, over
pixels with |D - D_A| or |D - D_B| < 0.03, and elsewhere |D - D_A|, |D - D_B| > 0.2) and saves the panels and a strip.

  two_focal_planes.py <aif.png> <xA> <yA> <xB> <yB> <strength> <out_prefix>"""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_utils import add_label, laplacian_sharpness, mark_point, save_rgb
from flashrefocus import Refocuser
from flashrefocus.rendering import depth_edits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aif", type=Path)
    ap.add_argument("point_a", type=float, nargs=2)
    ap.add_argument("point_b", type=float, nargs=2)
    ap.add_argument("strength", type=float)
    ap.add_argument("out_prefix")
    a = ap.parse_args()
    rf = Refocuser.from_pretrained(deblurring=False)
    scene = rf.scene_from_file(a.aif)
    da, db = scene.focal_depth_at(*a.point_a), scene.focal_depth_at(*a.point_b)
    to_array = lambda x: x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    aif, depth = to_array(scene.image), scene.depth[0].cpu().numpy()
    two_planes = depth_edits.two_focal_planes(scene.depth, da, db)
    renders = {"focus A": to_array(rf.render(scene, a.strength, da)), "focus B": to_array(rf.render(scene, a.strength, db)),
               "two planes": to_array(rf.render(scene, a.strength, da, two_planes))}

    near_a, near_b = np.abs(depth - da) < 0.03, np.abs(depth - db) < 0.03
    elsewhere = (np.abs(depth - da) > 0.2) & (np.abs(depth - db) > 0.2)
    relative = lambda img, mask: laplacian_sharpness(img, mask) / laplacian_sharpness(aif, mask)
    for name, img in renders.items():
        row = {"sharp_A": relative(img, near_a), "sharp_B": relative(img, near_b),
               "sharp_elsewhere": relative(img, elsewhere) if elsewhere.sum() > 100 else None}
        print(name, {k: (round(v, 3) if v is not None else None) for k, v in row.items()})

    out = Path(a.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    mark_a = lambda img: mark_point(img, *a.point_a)
    mark_both = lambda img: mark_point(mark_a(img), *a.point_b)
    depth_view = np.repeat((1 - two_planes[0].cpu().numpy())[..., None], 3, -1)
    panels = {"input": aif, "A": mark_a(renders["focus A"]), "B": mark_point(renders["focus B"], *a.point_b),
              "dual": mark_both(renders["two planes"]), "depth": depth_view}
    save_rgb(np.concatenate([add_label(panels["input"], "input"), add_label(panels["A"], "focus A"),
                             add_label(panels["B"], "focus B"), add_label(panels["dual"], "two focal planes"),
                             add_label(depth_view, "edited depth")], 1), out.with_suffix(".jpg"))
    for name, img in panels.items():
        save_rgb(img, Path(f"{out}_{name}.png"))
    for name, img in (("input", aif), ("A", renders["focus A"]), ("B", renders["focus B"]), ("dual", renders["two planes"])):
        save_rgb(img, Path(f"{out}_{name}_raw.png"))
    Path(f"{out}_points.json").write_text(json.dumps({"A": list(a.point_a), "B": list(a.point_b)}))


if __name__ == "__main__":
    main()
