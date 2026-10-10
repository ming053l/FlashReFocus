#!/usr/bin/env python
"""LVCorr: how well the blur-strength control tracks the rendered blur.

Following GenRefocus (arXiv:2512.16923, Sec. 4.1): for each all-in-focus image and a fixed focal point, render at ten
strengths spanning the method's range, measure the Laplacian variance of every output, and take the Pearson correlation
between the strength and the blur level (negative Laplacian variance); 1 means blur grows exactly with the strength.
Reported as the mean over images. Inputs: <bench>/aif/<key>.png and <bench>/focus_points.json (LFDOF-Refocus, 50
scenes, or 3CReal). Other methods are rendered with their own code on the same inputs and strength grid."""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
paths.add_to_path()

BENCH = paths.WORK / "lfdof_refocus"
OUT = paths.WORK / "lvcorr"


def lapvar(img_rgb_uint8):
    g = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def flashrefocus_backend():
    from infer_single_focus import BokehliciousFocusEngine, parse_args
    b = parse_args(["--input", "x.png", "--output", "y.png", "--focus-x", "0", "--focus-y", "0", "--bokeh-strength", "0.5"]
                   + (["--amp"] if os.environ.get("RENDER_BF16", "1") == "1" else []))   # bf16 autocast by default
    eng = BokehliciousFocusEngine(checkpoint=b.checkpoint, size=b.size, attn_version=b.attn_version, da2_root=b.da2_root,
                                  da2_checkpoint=b.da2_checkpoint, da2_encoder=b.da2_encoder, da2_input_size=b.da2_input_size,
                                  no_da2_amp=b.no_da2_amp, da2_amp_dtype=b.da2_amp_dtype, device=b.device, amp=b.amp)
    tmp = OUT / f"tmp_flashrefocus_{os.getpid()}.png"

    def render(path, fx, fy, k):
        eng.infer_image(path, tmp, focus_x=fx, focus_y=fy, bokeh_embedding=k / 100.0, marked_output=None,
                        resize_size=None, resize_short=False, coords_after_resize=False,
                        focus_map_type=b.focus_map_type, mark_size=b.mark_size)
        return cv2.cvtColor(cv2.imread(str(tmp)), cv2.COLOR_BGR2RGB)
    return render, list(np.linspace(10, 100, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["flashrefocus"], default="flashrefocus")
    ap.add_argument("--bench", type=Path, default=BENCH, help="folder with aif/NN.png and focus_points.json")
    ap.add_argument("--tag", default="", help="suffix of the output file, e.g. _3creal")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ann = json.loads((a.bench / "focus_points.json").read_text())["annotations"]
    render, ks = {"flashrefocus": flashrefocus_backend}[a.method]()
    per = {}
    for key in sorted(ann):
        path = a.bench / "aif" / f"{key}.png"
        lv = [lapvar(render(path, ann[key]["x"], ann[key]["y"], k)) for k in ks]
        per[key] = {"lv": lv, "lvcorr": pearson(ks, [-v for v in lv])}
        print(a.method, key, round(per[key]["lvcorr"], 4), flush=True)
    vals = [v["lvcorr"] for v in per.values() if np.isfinite(v["lvcorr"])]
    res = {"method": a.method, "bench": str(a.bench), "k_grid": [float(k) for k in ks], "lvcorr_mean": float(np.mean(vals)), "n": len(vals), "per_image": per}
    (OUT / f"{a.method}{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(a.method, "LVCorr", round(res["lvcorr_mean"], 4))


if __name__ == "__main__":
    main()
