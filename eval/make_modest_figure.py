#!/usr/bin/env python
"""Appendix figure: mean best strength of each method on the MODEST ladders, normalised by its f/2.8 value, against 1/N,
with the thin-lens reference (1/N - 1/22) / (1/2.8 - 1/22) for the extra blur relative to the f/22 input.
usage: make_modest_figure.py <out.pdf>"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = json.loads((Path(__file__).resolve().parent / "rbval_out/modest_summary.json").read_text())
N = np.array([16.0, 9.0, 5.0, 2.8]); x = 1 / N
STYLE = {"FlashReFocus": ("#2a78d6", "o"), "BokehMe": ("#eb6834", "s"), "DeepLens": ("#1baf7a", "^"), "MagicBokeh": ("#eda100", "D")}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ref = (x - 1 / 22) / (1 / 2.8 - 1 / 22)
ax.plot(x, ref, ls="--", color="#8a8a85", lw=1, zorder=1)
ax.text(x[1] + 0.004, ref[1] - 0.075, "thin lens", color="#5f5e58", fontsize=7)
for m, (c, mk) in STYLE.items():
    y = np.array(S[m]["mean_strength"]); y = y / y[-1]
    ax.plot(x, y, color=c, lw=2 if m == "FlashReFocus" else 1.3, marker=mk, ms=4.5, mec="white", mew=0.8, zorder=3 if m == "FlashReFocus" else 2)
lab = {"FlashReFocus": (0.215, 0.36), "BokehMe": (0.235, 0.20), "DeepLens": (0.26, 0.47), "MagicBokeh": (0.10, 0.62)}
for m, (lx, ly) in lab.items():
    ax.text(lx, ly, "FlashReFocus" if m == "FlashReFocus" else m, color="#1d1d1b", fontsize=7, weight="bold" if m == "FlashReFocus" else "normal")
    ax.plot([lx - 0.012], [ly + 0.02], marker=STYLE[m][1], color=STYLE[m][0], ms=4, mec="white", mew=0.6)
ax.set_xticks(x); ax.set_xticklabels(["f/16", "f/9", "f/5", "f/2.8"])
ax.set_xlabel("target aperture (spaced by 1/N)"); ax.set_ylabel("best strength / value at f/2.8")
ax.set_ylim(0, 1.08); ax.grid(axis="y", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig(sys.argv[1]); fig.savefig(Path(sys.argv[1]).with_suffix(".png"), dpi=200)
