#!/usr/bin/env python
"""LVCorr: how well a renderer's blur-strength control tracks the rendered blur.

Following GenRefocus (arXiv:2512.16923, Sec. 4.1 (ii), Table 3): for each all-in-focus
image and a fixed focus point, render at several strengths K spanning the method's
own range, measure the Laplacian variance (sharpness) of every output, and compute
the Pearson correlation between K and the blur level (negative Laplacian variance),
so 1 means blur grows exactly with K. Reported per method as the mean over images.

Inputs: LFDOF-Refocus all-in-focus images and focus points (50 scenes).
Backends: derefocus (our renderer), bokehme (BokehMe with Depth Anything V2 disparity).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

DEF = Path("/ssd1/chingheng/Defocus")
BENCH = DEF / "refocus_bench" / "lfdof_refocus"
OUT = DEF / "refocus_bench" / "lvcorr"


def lapvar(img_rgb_uint8):
    g = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def derefocus_backend():
    sys.path.insert(0, str(DEF / "SOTA" / "Bokehlicious"))
    os.chdir(DEF / "SOTA" / "Bokehlicious")
    from infer_single_focus import BokehliciousFocusEngine, parse_args
    b = parse_args(["--input", "x.png", "--output", "y.png", "--focus-x", "0", "--focus-y", "0", "--bokeh-strength", "0.5"]
                   + (["--amp"] if os.environ.get("RENDER_BF16", "1") == "1" else []))   # bf16 autocast, as the bokeh tables
    eng = BokehliciousFocusEngine(checkpoint=b.checkpoint, size=b.size, attn_version=b.attn_version, da2_root=b.da2_root,
                                  da2_checkpoint=b.da2_checkpoint, da2_encoder=b.da2_encoder, da2_input_size=b.da2_input_size,
                                  no_da2_amp=b.no_da2_amp, da2_amp_dtype=b.da2_amp_dtype, device=b.device, amp=b.amp)
    tmp = OUT / f"tmp_derefocus_{os.getpid()}.png"

    def render(path, fx, fy, k):
        eng.infer_image(path, tmp, focus_x=fx, focus_y=fy, bokeh_embedding=k / 100.0, marked_output=None,
                        resize_size=None, resize_short=False, coords_after_resize=False,
                        focus_map_type=b.focus_map_type, mark_size=b.mark_size)
        return cv2.cvtColor(cv2.imread(str(tmp)), cv2.COLOR_BGR2RGB)
    return render, list(np.linspace(10, 100, 10))


def bokehme_backend():
    sys.path.insert(0, str(DEF / "SOTA" / "BokehMe"))
    sys.path.insert(0, str(DEF / "SOTA" / "Depth-Anything-V2"))
    os.chdir(DEF / "SOTA" / "BokehMe")
    import test_ebb_midas_k_search as T
    from depth_anything_v2.dpt import DepthAnythingV2
    dev = torch.device("cuda")
    argv, sys.argv = sys.argv, sys.argv[:1]
    args = T.parse_args()
    sys.argv = argv
    args.arnet_checkpoint_path = str(DEF / "SOTA/BokehMe/checkpoints/arnet.pth")
    args.iunet_checkpoint_path = str(DEF / "SOTA/BokehMe/checkpoints/iunet.pth")
    renderer, arnet, iunet = T.load_bokehme(args, dev)
    da = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    da.load_state_dict(torch.load(DEF / "SOTA/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth", map_location="cpu"))
    da = da.to(dev).eval()
    cache = {}

    def render(path, fx, fy, k):
        if path not in cache:
            bgr = cv2.imread(str(path))
            disp = da.infer_image(bgr)
            disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
            img = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
            cache[path] = (img, torch.from_numpy(disp.astype(np.float32))[None, None].to(dev), disp)
        img, disp_t, disp = cache[path]
        focus = float(disp[int(fy), int(fx)])
        with torch.no_grad():
            out = T.render_pipeline(renderer, arnet, iunet, img, k * (disp_t - focus), args.gamma, args)
        return (out[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return render, list(np.linspace(10, 100, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["derefocus", "bokehme"])
    ap.add_argument("--bench", type=Path, default=BENCH, help="folder with aif/NN.png and focus_points.json")
    ap.add_argument("--tag", default="", help="suffix of the output file, e.g. _3creal")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ann = json.loads((a.bench / "focus_points.json").read_text())["annotations"]
    render, ks = {"derefocus": derefocus_backend, "bokehme": bokehme_backend}[a.method]()
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
