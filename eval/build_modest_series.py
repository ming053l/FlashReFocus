#!/usr/bin/env python
"""MODEST aperture series (arXiv 2511.20853): per scene, camera A at 70 mm, the first viewpoint with all five apertures.
The camera moves between frames, so for each f/22 frame the frame of every other aperture with the smallest
phase-correlation shift is taken, and the f/22 frame whose set aligns best is kept (at most 2 px at 1536). f/22 is the all-in-focus input, f/16, f/9, f/5 and f/2.8
are targets. Images are resized to a long side of 1536. The focal point is chosen automatically as the textured f/22 block
where the aligned f/2.8 and f/22 high-pass images correlate best (exposure-normalised), since MODEST has no focus annotation.
Writes modest_series/{original,bokeh_F16.0,bokeh_F9.0,bokeh_F5.0,bokeh_F2.8}/<scene>.jpg and focus_point.json."""
import glob
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from flashrefocus import paths

SRC = paths.DATA / "MODEST"
OUT = paths.WORK / "modest_series"
APS = ["F22.0", "F16.0", "F9.0", "F5.0", "F2.8"]
LONG = 1536
MAX_VIEWS = 5


def shift(a, b):
    """Phase-correlation shift (pixels at the working resolution) between two grayscale arrays."""
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b)); r = np.fft.ifft2(F / (np.abs(F) + 1e-9)).real
    y, x = np.unravel_index(np.argmax(r), r.shape)
    return (y if y < r.shape[0] // 2 else y - r.shape[0], x if x < r.shape[1] // 2 else x - r.shape[1])


def ncc(a, b):
    a = (a - a.mean()) / (a.std() + 1e-6); b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def main():
    for d in ["original"] + [f"bokeh_{a}" for a in APS[1:]]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    ann, report = {}, {}
    for s in range(1, 10):
        fs = sorted(glob.glob(str(SRC / f"Scene{s}/EOS6D_A_*/fl_70mm/*/*/IMG_*.JPG")), key=lambda f: int(re.search(r"IMG_(\d+)", f).group(1)))
        if not fs:
            continue
        ap = {f: Path(f).parent.name for f in fs}
        small768 = {f: np.asarray(Image.open(f).convert("L").resize((768, 512), Image.BOX)).astype(float) for f in fs}
        cands = {x: [f for f in fs if ap[f] == x] for x in APS}
        sets, used22 = [], []
        for f22 in cands["F22.0"]:
            g22 = small768[f22]
            if any(max(abs(np.array(shift(g22, small768[u])))) <= 1 for u in used22):
                continue                                  # another f/22 frame of an already used viewpoint
            sel, worst = {"F22.0": f22}, 0
            for x in APS[1:]:
                # same viewpoint: low-resolution correlation with f/22 >= 0.9 (phase correlation alone reports a
                # spurious zero shift between different views), then the smallest shift
                sh = [(abs(np.array(shift(g22, small768[f]))).max(), f) for f in cands[x] if ncc(g22, small768[f]) >= 0.9]
                if not sh:
                    break
                dd, f = min(sh); sel[x] = f; worst = max(worst, dd)
            else:
                if worst <= 1:
                    sets.append(sel); used22.append(f22)
        report[s] = {"aligned_viewpoints": len(sets), "kept": []}
        n = 0
        for sel in sets:
            if n >= MAX_VIEWS:
                break
            ims = {x: Image.open(sel[x]).convert("RGB") for x in APS}
            w, h = ims["F22.0"].size; sc = LONG / max(w, h); size = (round(w * sc), round(h * sc))
            ims = {x: im.resize(size, Image.LANCZOS) for x, im in ims.items()}
            g = {x: np.asarray(im.convert("L")).astype(float) for x, im in ims.items()}
            g = {x: v * (g["F22.0"].mean() / v.mean()) for x, v in g.items()}   # exposure normalisation
            hp = {x: g[x] - ndimage.gaussian_filter(g[x], 3) for x in ("F22.0", "F2.8")}
            k = 32
            mu = lambda v: ndimage.uniform_filter(v, k)
            a22, a28 = hp["F22.0"], hp["F2.8"]
            cov = mu(a22 * a28) - mu(a22) * mu(a28)
            v22 = mu(a22 * a22) - mu(a22) ** 2; v28 = mu(a28 * a28) - mu(a28) ** 2
            corr = cov / np.sqrt(np.clip(v22 * v28, 1e-6, None))
            m = 48; valid = np.zeros_like(corr, bool); valid[m:-m, m:-m] = True
            valid &= v22 > np.percentile(v22[m:-m, m:-m], 70)                    # textured regions only
            med = float(np.median(corr[valid]))
            if med < 0.3:                                 # not the same view (camera moved, or motion blur)
                report[s]["kept"].append({"f22": Path(sel["F22.0"]).name, "rejected_median_corr": med}); continue
            y, x = np.unravel_index(np.argmax(np.where(valid, corr, -2)), corr.shape)
            key = f"{s}{n:02d}"                           # e.g. 300 = scene 3, first kept viewpoint
            ims["F22.0"].save(OUT / "original" / f"{key}.jpg", quality=95)
            for a in APS[1:]:
                ims[a].save(OUT / f"bokeh_{a}" / f"{key}.jpg", quality=95)
            ann[key] = {"x": float(x), "y": float(y), "width": size[0], "height": size[1], "path": f"original/{key}.jpg"}
            report[s]["kept"].append({"key": key, "files": {a: Path(sel[a]).name for a in APS}, "focus": [int(x), int(y)],
                                      "focus_corr": float(corr[y, x]), "median_corr": med})
            n += 1
    json.dump({"coordinate_system": "pixels of the 1536-long-side images", "source": "MODEST camera A, 70 mm, first "
               "aligned viewpoints (up to MAX_VIEWS per scene); automatic focal point (see build_modest_series.py)", "annotations": ann},
              open(OUT / "focus_point.json", "w"), indent=1)
    json.dump(report, open(OUT / "build_report.json", "w"), indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
