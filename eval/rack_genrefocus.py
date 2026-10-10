#!/usr/bin/env python
"""Focus sweep with GenRefocus for the edit-consistency comparison: its bokeh stage (BokehNetEngine: Depth Pro depth,
K = 15, 28 steps, fixed seeds) on the same all-in-focus inputs and focal points as rack_ours.py.
Run in the GenRefocus environment, with its code under FLASHREFOCUS_BASELINES/Genfocus.
Usage: rack_genrefocus.py [<sweep_dir>] [<aif_dir>]"""
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
paths.add_to_path()

GEN = paths.BASELINES / "Genfocus"   # GenRefocus code (github.com/rayray9999/Genfocus)
os.chdir(GEN)
sys.path.insert(0, str(GEN))
import torch  # noqa: E402
import Inference_bokehNet as IB  # noqa: E402

FLUX = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-dev")   # Hugging Face id or a local copy
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else str(WORK_ROOT / "rack_focus"))
AIF = Path(sys.argv[2] if len(sys.argv) > 2 else str(WORK_ROOT / "3creal_lvcorr/aif"))
K = 15.0

eng = IB.BokehNetEngine(model_id=FLUX)
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
