#!/usr/bin/env python
"""Two focal planes by depth editing, without retraining: with focal depths d_A and d_B, the renderer receives
d' = d_A + min(|d - d_A|, |d - d_B|), so every pixel is blurred by its distance to the nearer of the two planes.
Renders plane A only, plane B only and both, measures the sharpness around each plane (mean |Laplacian| relative to the
input), and saves the panels and a strip.
Usage: dual_focus.py <aif.png> <xA> <yA> <xB> <yB> <strength> <out_prefix>"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
paths.add_to_path()
import fast_refocus as FR  # noqa: E402
import rack_analyze as RA  # noqa: E402

aif_p = Path(sys.argv[1]); xa, ya, xb, yb = map(float, sys.argv[2:6]); st = float(sys.argv[6]); out = Path(sys.argv[7])
FR.setup_backend(True)
rd = FR.FastRenderer()
rd.set_image(aif_p)
da, db = rd.depth_at(xa, ya), rd.depth_at(xb, yb)
aif = rd.src[0].permute(1, 2, 0).cpu().numpy()
dn = rd.depth[0].cpu().numpy()


def render(x, y, depth=None):
    return rd.render(x, y, st, depth=depth)[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def sharp(img, m):
    return float(np.abs(ndimage.laplace(img.mean(-1)))[m].mean())


A = render(xa, ya)
B = render(xb, yb)
dual_depth = rd.two_plane_depth((xa, ya), (xb, yb))
D = render(xa, ya, dual_depth)
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
