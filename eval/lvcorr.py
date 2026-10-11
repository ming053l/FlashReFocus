#!/usr/bin/env python
"""LVCorr: how well the blur-strength control tracks the rendered blur.

Following GenRefocus (arXiv:2512.16923, Sec. 4.1): for each all-in-focus image and a fixed focal point, render at ten
strengths spanning the method's range, measure the Laplacian variance of every output, and take the Pearson correlation
between the strength and the blur level (negative Laplacian variance); 1 means blur grows exactly with the strength.
Reported as the mean over images. Inputs: <bench>/aif/<key>.png and <bench>/focus_points.json (LFDOF-Refocus, 50
scenes, or 3CReal). Other methods are rendered with their own code on the same inputs and strength grid."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from flashrefocus import Refocuser, paths
from flashrefocus.image_io import tensor_to_uint8

BENCH = paths.WORK / "lfdof_refocus"
OUT = paths.WORK / "lvcorr"
STRENGTHS = np.linspace(10, 100, 10)   # in percent


def laplacian_variance(rgb_uint8):
    grey = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(grey, cv2.CV_32F).var())


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, default=BENCH, help="folder with aif/NN.png and focus_points.json")
    ap.add_argument("--tag", default="", help="suffix of the output file, e.g. _3creal")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rf = Refocuser.from_pretrained(deblurring=False)
    annotations = json.loads((a.bench / "focus_points.json").read_text())["annotations"]
    per_image = {}
    for key in sorted(annotations):
        scene = rf.scene_from_file(a.bench / "aif" / f"{key}.png")
        focal_depth = scene.focal_depth_at(annotations[key]["x"], annotations[key]["y"])
        lv = [laplacian_variance(tensor_to_uint8(rf.render(scene, k / 100.0, focal_depth))) for k in STRENGTHS]
        per_image[key] = {"lv": lv, "lvcorr": pearson(STRENGTHS, [-v for v in lv])}
        print(key, round(per_image[key]["lvcorr"], 4), flush=True)
    values = [v["lvcorr"] for v in per_image.values() if np.isfinite(v["lvcorr"])]
    res = {"method": "flashrefocus", "bench": str(a.bench), "k_grid": [float(k) for k in STRENGTHS],
           "lvcorr_mean": float(np.mean(values)), "n": len(values), "per_image": per_image}
    (OUT / f"flashrefocus{a.tag}.json").write_text(json.dumps(res, indent=1))
    print("LVCorr", round(res["lvcorr_mean"], 4))


if __name__ == "__main__":
    main()
