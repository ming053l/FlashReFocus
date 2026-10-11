#!/usr/bin/env python
"""Focal-edit study, setting (ii): GenRefocus matched to FlashReFocus at strength 0.5. Per scene, K is chosen by binary
search in [2, 15] so that GenRefocus' out-of-focus sharpness ratio (see focal_edits_match_ours.py) matches
FlashReFocus' at 0.5, estimated on three of the eight focal settings; all eight are then rendered at that K into
<scene>/genrefocus_matched. Run in the GenRefocus environment after focal_edits_match_ours.py.
Writes blur_match_genrefocus.json."""
import json
import os
import sys

import numpy as np

from flashrefocus import paths
from eval_utils import laplacian_sharpness, load_rgb
from focal_edits_common import AIF, OUT, OUT_OF_FOCUS

GENREFOCUS = paths.BASELINES / "Genfocus"   # github.com/rayray9999/Genfocus
FLUX = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
PROBE_SETTINGS = (0, 4, 7)


def main():
    os.chdir(GENREFOCUS)
    sys.path.insert(0, str(GENREFOCUS))
    import Inference_bokehNet
    engine = Inference_bokehNet.BokehNetEngine(model_id=FLUX)
    ours = json.loads((OUT / "blur_match_ours.json").read_text())
    probe_dir = OUT / "_genrefocus_probe"
    probe_dir.mkdir(exist_ok=True)
    result = {}
    for s, info in ours.items():
        scene_dir = OUT / s
        points = json.loads((scene_dir / "points.json").read_text())
        depth = np.load(scene_dir / "depth.npy")
        aif_path = AIF / f"{s}.png"
        aif = load_rgb(aif_path)
        size = (aif.shape[1], aif.shape[0])
        masks = [np.abs(depth - p["depth"]) > OUT_OF_FOCUS for p in points]
        base = [laplacian_sharpness(aif, m) for m in masks]
        goal = info["ours_ratio_at_0.5"]

        def ratio(k, indices, dst_dir):
            r = []
            for i in indices:
                dst = dst_dir / f"{i:02d}.png"
                engine.infer_image(aif_path, dst, k_value=float(k), point=f"{points[i]['x']},{points[i]['y']}")
                r.append(laplacian_sharpness(load_rgb(dst, size), masks[i]) / base[i])
            return float(np.mean(r))

        lo, hi, tried = 2.0, 15.0, {}                   # the ratio decreases with K
        for _ in range(4):
            mid = round((lo + hi) / 2, 2)
            tried[mid] = ratio(mid, PROBE_SETTINGS, probe_dir)
            lo, hi = (mid, hi) if tried[mid] > goal else (lo, mid)
        k = min(tried, key=lambda kk: abs(tried[kk] - goal))
        (scene_dir / "genrefocus_matched").mkdir(exist_ok=True)
        result[s] = {"target_ratio_ours_0.5": goal, "tried": tried, "k": k,
                     "genrefocus_ratio_all8": ratio(k, range(len(points)), scene_dir / "genrefocus_matched")}
        print(s, result[s], flush=True)
    (OUT / "blur_match_genrefocus.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
