#!/usr/bin/env python
"""GenRefocus focus sweep with its blur matched to FlashReFocus at strength 0.5: per scene, K is chosen by binary search
in [2, 15] so that GenRefocus' out-of-focus sharpness ratio (mean |Laplacian| over pixels far from the focal depth,
relative to the all-in-focus input; see rack_match_blur.py) matches FlashReFocus' at strength 0.5, estimated on three
of the eight focal settings; all eight settings are then rendered at that K into <scene>/genrefocus_matched/.
Run in the GenRefocus environment. Usage: rack_genrefocus_match.py <sweep_dir> <aif_dir>"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
paths.add_to_path()
OUT = Path(sys.argv[1]); AIF = Path(sys.argv[2])
GEN = paths.BASELINES / "Genfocus"   # GenRefocus code (github.com/rayray9999/Genfocus)
FLUX = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-dev")   # Hugging Face id or a local copy
os.chdir(GEN); sys.path.insert(0, str(GEN))
import Inference_bokehNet as IB  # noqa: E402

eng = IB.BokehNetEngine(model_id=FLUX)
match = json.loads((OUT / "blur_match.json").read_text())
TMP = OUT / "_kmatch_tmp"; TMP.mkdir(exist_ok=True)


def load(p, size):
    im = Image.open(p).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def sharp(img, m):
    return float(np.abs(ndimage.laplace(img.mean(-1)))[m].mean())


result = {}
for s, info in match.items():
    o = OUT / s; pts = json.loads((o / "points.json").read_text()); d = np.load(o / "depth.npy")
    aif_p = AIF / f"{s}.png"; aif = load(aif_p, Image.open(aif_p).size); size = (aif.shape[1], aif.shape[0])
    masks = [np.abs(d - p["depth"]) > 0.25 for p in pts]; base = [sharp(aif, m) for m in masks]
    target = info["ours_ratio_at_0.5"]
    probe = (0, 4, 7)

    def ratio(k, idx, dst_dir):
        r = []
        for i in idx:
            dst = dst_dir / f"{i:02d}.png"
            eng.infer_image(aif_p, dst, k_value=float(k), point=f"{pts[i]['x']},{pts[i]['y']}")
            r.append(sharp(load(dst, size), masks[i]) / base[i])
        return float(np.mean(r))

    lo, hi = 2.0, 15.0                                  # ratio decreases with K
    tried = {}
    for _ in range(4):
        mid = round((lo + hi) / 2, 2)
        tried[mid] = ratio(mid, probe, TMP)
        if tried[mid] > target:
            lo = mid                                    # still sharper than ours: more blur
        else:
            hi = mid
    k = min(tried, key=lambda kk: abs(tried[kk] - target))
    (o / "genrefocus_matched").mkdir(exist_ok=True)
    full = ratio(k, range(len(pts)), o / "genrefocus_matched")
    result[s] = {"target_ratio_ours_0.5": target, "tried": tried, "k": k, "genrefocus_ratio_all8": full}
    print(s, result[s], flush=True)
(OUT / "blur_match_genrefocus.json").write_text(json.dumps(result, indent=1))
