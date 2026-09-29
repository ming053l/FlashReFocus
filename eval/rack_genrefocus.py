#!/usr/bin/env python
"""GenRefocus bokeh stage (official BokehNetEngine: Depth Pro depth, K=15, 28 steps, fixed seeds) on the same AiF inputs
and focus points as rack_ours.py, for the rack-focus consistency comparison. Run from SOTA/Genfocus in the Genfocus env."""
import json
import os
import sys
import time
from pathlib import Path

GEN = Path("/ssd1/chingheng/Defocus/SOTA/Genfocus")
os.chdir(GEN)
sys.path.insert(0, str(GEN))
import torch  # noqa: E402
import Inference_bokehNet as IB  # noqa: E402

LOCAL_FLUX = "/ssd2/ming/Genfocus/models/FLUX.1-dev"   # the gated HF id is not accessible from this account
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/ssd1/chingheng/Defocus/refocus_bench/rack_focus")
AIF = Path(sys.argv[2] if len(sys.argv) > 2 else "/ssd1/chingheng/Defocus/refocus_bench/3creal_lvcorr/aif")
K = 15.0

eng = IB.BokehNetEngine(model_id=LOCAL_FLUX)
times = {}
for scene_dir in sorted(p for p in OUT.iterdir() if (p / "points.json").exists()):
    pts = json.loads((scene_dir / "points.json").read_text())
    (scene_dir / "genrefocus").mkdir(exist_ok=True)
    ts = []
    for i, p in enumerate(pts):
        dst = scene_dir / "genrefocus" / f"{i:02d}.png"
        if dst.exists():
            continue
        torch.cuda.synchronize(); t0 = time.perf_counter()
        eng.infer_image(AIF / f"{scene_dir.name}.png", dst, k_value=K, point=f"{p['x']},{p['y']}")
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    times[scene_dir.name] = ts
(OUT / "meta_genrefocus.json").write_text(json.dumps({"k": K, "seconds_per_frame": times}, indent=1))
print("done", {k: (sum(v) / len(v) if v else None) for k, v in times.items()})
