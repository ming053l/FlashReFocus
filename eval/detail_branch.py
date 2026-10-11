#!/usr/bin/env python
"""Where the Stage I detail branch acts (Appendix F, Fig. F.4): on the DPDD test pairs, the difference between the
outputs at omega = 1.25 and omega = 0 (pixel branch only), its share on the 20% of pixels with the most texture in the
sharp target (local Sobel energy, 7x7 mean) and the error to the target on those pixels and on the smoothest half."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
paths.add_to_path()
import fast_refocus as FR  # noqa: E402
from model import read_image  # noqa: E402

SOBEL = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def texture(img):
    """Local detail energy of a 3 x H x W image in [0, 1]."""
    g = F.pad(img.mean(0)[None, None], (1,) * 4, mode="replicate")
    s = SOBEL.to(img)
    mag = torch.sqrt(F.conv2d(g, s.view(1, 1, 3, 3)) ** 2 + F.conv2d(g, s.t().reshape(1, 1, 3, 3)) ** 2)
    return F.avg_pool2d(mag, 7, 1, 3)[0, 0]


@torch.no_grad()
def main():
    pairs = json.loads((paths.REPO / "deblur" / "pairs" / "dpdd_test.json").read_text())
    FR.setup_backend(False)
    s1 = FR.FastStageI()
    rows = []
    for p in pairs:
        x = read_image(paths.DATA / p["source"]).cuda()
        target = read_image(paths.DATA / p["target"]).cuda()[0] * 0.5 + 0.5
        out = {}
        for omega in (0.0, 1.25):
            s1.omega = omega
            out[omega] = s1.run_tensor(x).float().clamp(-1, 1)[0] * 0.5 + 0.5
        diff = (out[1.25] - out[0.0]).abs().mean(0)
        t = texture(target)
        textured, smooth = t >= torch.quantile(t.flatten()[::7], 0.8), t <= torch.quantile(t.flatten()[::7], 0.5)
        err = {o: (out[o] - target).abs().mean(0) * 255 for o in out}
        rows.append([float(diff.mean() * 255), float(diff[textured].sum() / diff.sum()),
                     float(err[0.0][textured].mean()), float(err[1.25][textured].mean()),
                     float(err[0.0][smooth].mean()), float(err[1.25][smooth].mean())])
    r = np.mean(rows, axis=0)
    print(f"{len(rows)} images: mean |difference| {r[0]:.2f}/255, share on the most textured 20% {r[1]:.3f}; "
          f"error on textured pixels {r[2]:.2f} -> {r[3]:.2f}, on the smoothest half {r[4]:.2f} -> {r[5]:.2f} (/255)")


if __name__ == "__main__":
    main()
