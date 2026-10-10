#!/usr/bin/env python
"""Focus sweep with FlashReFocus on 3CReal all-in-focus inputs (long side 1024): picks the --n_scenes scenes with the
widest depth spread and, per scene, eight focal points at depth quantiles 95% .. 5% (far -> near), shared with
GenRefocus; renders the eight frames at --strength, a --video_frames sweep along a path through the points and a
strength sweep 0 -> 1, and saves the normalised depth (depth.npy) used for the in-focus masks. Logs the time per edit."""
import sys
import argparse
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
paths.add_to_path()
DATA_ROOT, WORK_ROOT = paths.DATA, paths.WORK

import numpy as np
import torch

import fast_refocus as FR

AIF = (WORK_ROOT / "3creal_lvcorr/aif")


def quantile_points(depth, qs, grid=24):
    """For each depth quantile, the grid pixel whose depth is closest (depth: H x W in [0, 1], 1 = far)."""
    h, w = depth.shape
    ys = np.linspace(0.08 * h, 0.92 * h, grid).astype(int)
    xs = np.linspace(0.08 * w, 0.92 * w, grid).astype(int)
    cand = [(int(x), int(y), float(depth[y, x])) for y in ys for x in xs]
    pts = []
    for q in qs:
        target = float(np.quantile(depth, q))
        x, y, d = min(cand, key=lambda c: abs(c[2] - target))
        pts.append({"x": x, "y": y, "depth": d, "quantile": float(q)})
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=(WORK_ROOT / "rack_focus"))
    ap.add_argument("--aif_dir", type=Path, default=AIF)
    ap.add_argument("--n_scenes", type=int, default=3)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--video_frames", type=int, default=60)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    FR.setup_backend(True)
    rd = FR.FastRenderer()

    spread = []
    for f in sorted(a.aif_dir.glob("*.png")):
        rd.set_image(f)
        d = rd.depth[0].cpu().numpy()
        spread.append((float(np.quantile(d, 0.95) - np.quantile(d, 0.05)), f.stem))
    scenes = [s for _, s in sorted(spread, reverse=True)[:a.n_scenes]]
    qs = np.linspace(0.95, 0.05, 8)                               # renderer depth: 1 = far, so far -> near
    meta = {"strength": a.strength, "aif_dir": str(a.aif_dir), "scenes": {}}
    for s in scenes:
        o = a.out / s; (o / "ours").mkdir(parents=True, exist_ok=True); (o / "video").mkdir(exist_ok=True)
        (o / "strength").mkdir(exist_ok=True)
        rd.set_image(a.aif_dir / f"{s}.png")
        d = rd.depth[0].cpu().numpy(); np.save(o / "depth.npy", d)
        pts = quantile_points(d, qs)
        ts = []
        for i, p in enumerate(pts):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            rd.render(p["x"], p["y"], a.strength, o / "ours" / f"{i:02d}.png")
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        # rack focus: straight-line path through the 8 points
        path = []
        seg = a.video_frames // (len(pts) - 1)
        for i in range(len(pts) - 1):
            for t in np.linspace(0, 1, seg, endpoint=False):
                path.append((pts[i]["x"] * (1 - t) + pts[i + 1]["x"] * t, pts[i]["y"] * (1 - t) + pts[i + 1]["y"] * t))
        path.append((pts[-1]["x"], pts[-1]["y"]))
        for i, (x, y) in enumerate(path):
            rd.render(x, y, a.strength, o / "video" / f"{i:03d}.png")
        mid = pts[len(pts) // 2]
        for i, st in enumerate(np.linspace(0, 1, 31)):
            rd.render(mid["x"], mid["y"], float(st), o / "strength" / f"{i:03d}.png")
        meta["scenes"][s] = {"points": pts, "depth_spread": dict((n, v) for v, n in spread)[s],
                             "render_s_mean": float(np.mean(ts[1:])), "video_path": path, "strength_focus": mid}
        (o / "points.json").write_text(json.dumps(pts, indent=1))
    (a.out / "meta_ours.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({s: meta["scenes"][s]["render_s_mean"] for s in scenes}))


if __name__ == "__main__":
    main()
