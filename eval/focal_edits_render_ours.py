#!/usr/bin/env python
"""Focal-edit study, FlashReFocus renders: on all-in-focus inputs (long side 1024), picks the --n_scenes scenes with
the widest depth spread and, per scene, eight focal points at depth quantiles 95% .. 5% (far -> near), shared with
GenRefocus (focal_edits_render_genrefocus.py). Renders the eight edits at --strength (<scene>/ours), a rack-focus video
along a path through the points (<scene>/video) and a strength sweep 0 -> 1 (<scene>/strength), and saves the renderer
depth (depth.npy) used for the masks. Logs the time per edit."""
import argparse
import json
import time

import numpy as np
import torch

from flashrefocus import Refocuser
from focal_edits_common import AIF, OUT


def quantile_points(depth, quantiles, grid=24):
    """For each depth quantile, the grid pixel whose depth is closest (depth: H x W in [0, 1], 1 = far)."""
    h, w = depth.shape
    ys = np.linspace(0.08 * h, 0.92 * h, grid).astype(int)
    xs = np.linspace(0.08 * w, 0.92 * w, grid).astype(int)
    candidates = [(int(x), int(y), float(depth[y, x])) for y in ys for x in xs]
    points = []
    for q in quantiles:
        target = float(np.quantile(depth, q))
        x, y, d = min(candidates, key=lambda c: abs(c[2] - target))
        points.append({"x": x, "y": y, "depth": d, "quantile": float(q)})
    return points


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_scenes", type=int, default=3)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--video_frames", type=int, default=60)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True    # fastest per-edit time, which this script reports
    rf = Refocuser.from_pretrained(deblurring=False)

    spread = {}
    for f in sorted(AIF.glob("*.png")):
        d = rf.scene_from_file(f).depth[0].cpu().numpy()
        spread[f.stem] = float(np.quantile(d, 0.95) - np.quantile(d, 0.05))
    scenes = sorted(spread, key=spread.get, reverse=True)[:a.n_scenes]
    meta = {"strength": a.strength, "aif_dir": str(AIF), "scenes": {}}
    for s in scenes:
        o = OUT / s
        for sub in ("ours", "video", "strength"):
            (o / sub).mkdir(parents=True, exist_ok=True)
        scene = rf.scene_from_file(AIF / f"{s}.png")
        depth = scene.depth[0].cpu().numpy()
        np.save(o / "depth.npy", depth)
        points = quantile_points(depth, np.linspace(0.95, 0.05, 8))
        times = []
        for i, p in enumerate(points):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            rf.render_to_file(scene, a.strength, scene.focal_depth_at(p["x"], p["y"]), o / "ours" / f"{i:02d}.png")
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        # rack focus: straight-line path through the eight points
        path, per_segment = [], a.video_frames // (len(points) - 1)
        for p, q in zip(points, points[1:]):
            for t in np.linspace(0, 1, per_segment, endpoint=False):
                path.append((p["x"] * (1 - t) + q["x"] * t, p["y"] * (1 - t) + q["y"] * t))
        path.append((points[-1]["x"], points[-1]["y"]))
        for i, (x, y) in enumerate(path):
            rf.render_to_file(scene, a.strength, scene.focal_depth_at(x, y), o / "video" / f"{i:03d}.png")
        mid = points[len(points) // 2]
        for i, k in enumerate(np.linspace(0, 1, 31)):
            rf.render_to_file(scene, float(k), scene.focal_depth_at(mid["x"], mid["y"]), o / "strength" / f"{i:03d}.png")
        meta["scenes"][s] = {"points": points, "depth_spread": spread[s], "render_s_mean": float(np.mean(times[1:])),
                             "video_path": path, "strength_focus": mid}
        (o / "points.json").write_text(json.dumps(points, indent=1))
    (OUT / "meta_ours.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({s: meta["scenes"][s]["render_s_mean"] for s in scenes}))


if __name__ == "__main__":
    main()
