#!/usr/bin/env python
"""Tilt-shift by depth editing, without retraining: the renderer receives a vertical depth ramp with a flat band
instead of the estimated depth, so a horizontal band stays sharp and blur grows above and below it.

Usage: tiltshift.py <input> <out_prefix> [--row 0.3] [--slope 3.0] [--band 0.06] [--strength 0.6] [--aif]
--row is the band centre as a fraction of the height; --aif skips Stage I for an image that is already all-in-focus.
Writes <out_prefix>_aif.png, _tiltshift.png (edited depth), _depthbokeh.png (estimated depth, focus on the band
centre) and _tiltdepth.png.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
paths.add_to_path()
import fast_refocus as FR  # noqa: E402


def fit(img, long_side=1024):
    """Long side at most `long_side`, both sides multiples of 16."""
    s = min(1.0, long_side / max(img.size))
    return img.resize((max(16, int(img.width * s) // 16 * 16), max(16, int(img.height * s) // 16 * 16)), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("out_prefix")
    ap.add_argument("--row", type=float, default=0.3)
    ap.add_argument("--slope", type=float, default=3.0)
    ap.add_argument("--band", type=float, default=0.06, help="half-height of the fully sharp band")
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--aif", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_prefix); out.parent.mkdir(parents=True, exist_ok=True)
    aif = Path(f"{out}_aif.png")
    if a.aif:
        shutil.copy(a.input, aif)
    else:
        src = Path(f"{out}_input.png")
        fit(Image.open(a.input).convert("RGB")).save(src)
        FR.FastStageI()(src, aif)
    rd = FR.FastRenderer()
    rd.set_image(aif)
    _, H, W = rd.depth.shape
    rd.render(W / 2, a.row * H, a.strength, Path(f"{out}_depthbokeh.png"))
    tilt = rd.tilt_shift_depth(a.row, a.slope, a.band)
    rd.render(W / 2, a.row * H, a.strength, Path(f"{out}_tiltshift.png"), depth=tilt)
    Image.fromarray((tilt[0].cpu().numpy() * 255).astype(np.uint8)).save(f"{out}_tiltdepth.png")


if __name__ == "__main__":
    main()
