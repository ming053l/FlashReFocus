#!/usr/bin/env python
"""Inside the renderer (Appendix C.1, Fig. C.2).

  renderer_internals.py curves     learned defocus embedding phi_D(M) and strength scale gamma_K(K) of every head
  renderer_internals.py residual   mean |Delta I| against the defocus magnitude M on 3CReal (long side 1024, annotated
                                   focal point, K = 0.25, 0.5, 0.75, 1; bf16 as in all results)
Both write .npz files under <work>/renderer_internals/."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
paths.add_to_path()
from method.blocks import DefocusBiasAttention  # noqa: E402

OUT = paths.WORK / "renderer_internals"


@torch.no_grad()
def curves():
    from method.model import load_renderer
    model = load_renderer(paths.RENDERER)
    m = torch.linspace(0, 1, 201, device="cuda").view(1, 1, 1, -1)
    k = torch.linspace(0, 1, 101, device="cuda").view(-1, 1)
    res = {"M": m.flatten().cpu().numpy(), "K": k.flatten().cpu().numpy()}
    for i, att in enumerate(a for a in model.modules() if isinstance(a, DefocusBiasAttention)):
        res[f"phi_D_{i}"] = att.defocus_head_proj(att._log_features(m))[0, 0].cpu().numpy()              # M, heads
        gamma_b = torch.exp(torch.clamp(att.aperture_scale_proj(k), -4, 4))
        res[f"gamma_K_{i}"] = (gamma_b * att.defocus_alpha).cpu().numpy()                               # K, heads
    np.savez(OUT / "curves.npz", **res)


@torch.no_grad()
def residual(root):
    import fast_refocus as FR
    ann = json.loads((root / "focus_point.json").read_text())["annotations"]
    FR.setup_backend(False)
    rd = FR.FastRenderer()
    strengths, edges = [0.25, 0.5, 0.75, 1.0], np.linspace(0, 1, 21)
    sums, count = np.zeros((len(strengths), 20)), np.zeros(20)
    tmp = OUT / "input.png"
    for key in sorted(ann, key=int):
        a = ann[key]
        im = Image.open(root / "original" / f"{key}.jpg").convert("RGB")
        s = 1024 / max(im.size)
        w, h = int(im.size[0] * s) // 16 * 16, int(im.size[1] * s) // 16 * 16
        im.resize((w, h), Image.BICUBIC).save(tmp)
        rd.set_image(tmp)
        fx, fy = a["x"] * w / a["width"], a["y"] * h / a["height"]
        x, y = max(0, min(w - 1, int(round(fx)))), max(0, min(h - 1, int(round(fy))))
        bins = np.clip(np.digitize((rd.depth[0] - rd.depth[0, y, x]).abs().cpu().numpy(), edges) - 1, 0, 19).ravel()
        count += np.bincount(bins, minlength=20)
        for i, k in enumerate(strengths):
            delta = (rd.render(fx, fy, k).float().clamp(0, 1) - rd.src).abs().mean(1)[0].cpu().numpy() * 255
            sums[i] += np.bincount(bins, weights=delta.ravel(), minlength=20)
    np.savez(OUT / "residual.npz", edges=edges, mean=sums / np.maximum(count, 1), count=count, K=np.array(strengths))
    print("mean |Delta I| (/255) for M < 0.05:", np.round(sums[:, 0] / max(count[0], 1), 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["curves", "residual"])
    ap.add_argument("--root", type=Path, default=paths.DATA / "3CReal" / "validation")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    curves() if a.what == "curves" else residual(a.root)


if __name__ == "__main__":
    main()
