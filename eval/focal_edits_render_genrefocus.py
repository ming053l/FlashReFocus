#!/usr/bin/env python
"""Focal-edit study, GenRefocus renders: its bokeh stage (BokehNetEngine: Depth Pro depth, K = 15, 28 steps, fixed
seeds) on the same all-in-focus inputs and focal points as focal_edits_render_ours.py, into <scene>/genrefocus.
Run in the GenRefocus environment, with its code under FLASHREFOCUS_BASELINES/Genfocus (FLUX_MODEL: a Hugging Face id
or a local copy of FLUX.1-dev)."""
import json
import os
import sys
import time

import torch

from flashrefocus import paths
from focal_edits_common import AIF, OUT, scene_dirs

GENREFOCUS = paths.BASELINES / "Genfocus"   # github.com/rayray9999/Genfocus
FLUX = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
K = 15.0


def main():
    os.chdir(GENREFOCUS)
    sys.path.insert(0, str(GENREFOCUS))
    import Inference_bokehNet
    engine = Inference_bokehNet.BokehNetEngine(model_id=FLUX)
    times = {}
    for scene in scene_dirs():
        points = json.loads((scene / "points.json").read_text())
        (scene / "genrefocus").mkdir(exist_ok=True)
        times[scene.name] = []
        for i, p in enumerate(points):
            dst = scene / "genrefocus" / f"{i:02d}.png"
            if dst.exists():
                continue
            torch.cuda.synchronize(); t0 = time.perf_counter()
            engine.infer_image(AIF / f"{scene.name}.png", dst, k_value=K, point=f"{p['x']},{p['y']}")
            torch.cuda.synchronize(); times[scene.name].append(time.perf_counter() - t0)
    (OUT / "meta_genrefocus.json").write_text(json.dumps({"k": K, "seconds_per_frame": times}, indent=1))


if __name__ == "__main__":
    main()
