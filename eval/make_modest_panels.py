#!/usr/bin/env python
"""Appendix panels for MODEST ladders: real captures (f/22 input, f/16, 9, 5, 2.8) and FlashReFocus renders at the
per-image best strength, plus the renderer depth (near bright) with the focal point; writes JPEGs to the paper's
figures/panels/ and prints the strength / SSIM labels. Usage: make_modest_panels.py <ladder key>..."""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import fast_refocus as FR

B = Path(__file__).resolve().parent; E = B / "modest_eval"; OUT = Path("/ssd1/chingheng/overleaf_6ab3c4e8/figures/panels")
APS = ["F16.0", "F9.0", "F5.0", "F2.8"]; W = 640
res = {a: {r["key"]: r for r in csv.DictReader(open(B / f"rbval_out/modest_ours_bf16_{a}/results.csv"))} for a in APS}
fp = json.loads((E / "F2.8/focus_point.json").read_text())["annotations"]
FR.setup_backend(True); rd = FR.FastRenderer()


def panel(im, name, xy=None):
    im = im.convert("RGB"); s = W / im.width; im = im.resize((W, round(im.height * s)), Image.LANCZOS)
    if xy is not None:
        d = ImageDraw.Draw(im); x, y = xy[0] * s, xy[1] * s; r = 11
        for w, c in ((6, (0, 0, 0)), (3, (0, 255, 0))):
            d.ellipse([x - r, y - r, x + r, y + r], outline=c, width=w)
    im.save(OUT / f"{name}.jpg", quality=92)


for k in sys.argv[1:]:
    src = Path(glob.glob(str(E / f"F2.8/original/{k}.*"))[0]); xy = (fp[k]["x"], fp[k]["y"])
    panel(Image.open(src), f"modest_{k}_input", xy)
    rd.set_image(src); dep = rd.depth[0].float().cpu().numpy()
    panel(Image.fromarray((np.clip(1 - dep, 0, 1) * 255).astype(np.uint8)), f"modest_{k}_depth", xy)
    labels = []
    for a in APS:
        panel(Image.open(glob.glob(str(E / f"{a}/bokeh/{k}.*"))[0]), f"modest_{k}_gt_{a}")
        panel(Image.open(B / f"rbval_out/modest_ours_bf16_{a}/images/{k}.png"), f"modest_{k}_ours_{a}")
        r = res[a][k]; labels.append((float(r["strength"]), float(r["ssim"])))
    print(k, labels)
