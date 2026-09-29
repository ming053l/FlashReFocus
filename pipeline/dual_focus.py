#!/usr/bin/env python
"""Two focal planes by depth editing (no retraining): with focal depths d_A (near subject) and d_B (far subject), the
renderer receives d' = f + min(|d - d_A|, |d - d_B|) with f = d_A, so every pixel's defocus is its distance to the nearer
of the two planes. Renders single-plane A, single-plane B and dual-plane results, measures the sharpness (mean |Laplacian|
relative to the input) around each plane, and saves a strip and the panels.
Usage: dual_focus.py <aif.png> <xA> <yA> <xB> <yB> <strength> <out_prefix>"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

import fast_refocus as FR
import rack_analyze as RA

aif_p = Path(sys.argv[1]); xa, ya, xb, yb = map(float, sys.argv[2:6]); st = float(sys.argv[6]); out = Path(sys.argv[7])
FR.setup_backend(True)
rd = FR.FastRenderer()
rd.set_image(aif_p)
d0 = rd.depth.clone()
da = float(d0[0, int(ya), int(xa)]); db = float(d0[0, int(yb), int(xb)])
aif = rd.src[0].permute(1, 2, 0).cpu().numpy()
dn = d0[0].cpu().numpy()


def render(depth, x, y):
    rd.depth = depth
    img = rd.render(x, y, st, None)[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    rd.depth = d0.clone()
    return img


def sharp(img, m):
    return float(np.abs(ndimage.laplace(img.mean(-1)))[m].mean())


A = render(d0.clone(), xa, ya)
B = render(d0.clone(), xb, yb)
dual_depth = (da + torch.minimum((d0 - da).abs(), (d0 - db).abs())).clamp(0, 1)
D = render(dual_depth, xa, ya)
ma, mb = np.abs(dn - da) < 0.03, np.abs(dn - db) < 0.03
far = (np.abs(dn - da) > 0.2) & (np.abs(dn - db) > 0.2)
rows = {}
for name, img in (("focus A", A), ("focus B", B), ("dual A+B", D)):
    rows[name] = {"sharp_A": sharp(img, ma) / sharp(aif, ma), "sharp_B": sharp(img, mb) / sharp(aif, mb),
                  "sharp_elsewhere": sharp(img, far) / sharp(aif, far) if far.sum() > 100 else None}
    print(name, {k: (round(v, 3) if v is not None else None) for k, v in rows[name].items()})
dd = dual_depth[0].cpu().numpy()
depth_vis = np.repeat((1 - dd)[..., None], 3, -1)
mark = lambda img: RA.mark(RA.mark(img, xa, ya), xb, yb)
RA.save(np.concatenate([RA.label(aif, "input"), RA.label(RA.mark(A, xa, ya), "focus A"), RA.label(RA.mark(B, xb, yb), "focus B"),
                        RA.label(mark(D), "two focal planes"), RA.label(depth_vis, "edited depth")], 1), out.with_suffix(".jpg"))
for name, img in (("input", aif), ("A", RA.mark(A, xa, ya)), ("B", RA.mark(B, xb, yb)), ("dual", mark(D)), ("depth", depth_vis)):
    RA.save(img, Path(f"{out}_{name}.png"))
for name, img in (("input", aif), ("A", A), ("B", B), ("dual", D)):
    RA.save(img, Path(f"{out}_{name}_raw.png"))
Path(f"{out}_points.json").write_text(json.dumps({"A": [xa, ya], "B": [xb, yb]}))
