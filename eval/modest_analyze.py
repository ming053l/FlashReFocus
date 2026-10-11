#!/usr/bin/env python
"""Strength controllability on MODEST aperture series: for each method, the per-image best strength (in the method's
own unit) found against the f/16, 9, 5 and 2.8 targets is compared with 1/N: the share of series whose best strength
never decreases as the aperture opens, the mean Spearman rank correlation per series, and the Pearson correlation of the
mean strength with 1/N; plus the mean SSIM per aperture. Reads <work>/rbval_out/modest_*; writes modest_summary.json."""
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

O = Path(os.environ.get("FLASHREFOCUS_WORK", Path(__file__).resolve().parents[1] / "work")) / "rbval_out"
APS = ["F16.0", "F9.0", "F5.0", "F2.8"]
INV = np.array([1 / float(a[1:]) for a in APS])
SRC = {"FlashReFocus": ("modest_ours_bf16_{}/results.csv", "key", "strength", "ssim"),
       "BokehMe": ("modest_bokehme_{}/results.csv", "scene_id", "best_k", "ssim"),
       "MagicBokeh": ("modest_magicbokeh_{}/metrics_per_image_shard0.csv", "key", "k", "ssim"),
       "DeepLens": ("modest_deeplens_{}/metrics.csv", "image", "aperture", "ssim")}
out = {}
for m, (pat, kc, sc, qc) in SRC.items():
    tab = {}
    for i, a in enumerate(APS):
        for r in csv.DictReader(open(O / pat.format(a))):
            tab.setdefault(r[kc].split(".")[0], [None] * 4)[i] = (float(r[sc]), float(r[qc]))
    keys = sorted(k for k in tab if all(tab[k]))
    S = np.array([[tab[k][i][0] for i in range(4)] for k in keys]); Q = np.array([[tab[k][i][1] for i in range(4)] for k in keys])
    mono = int(sum(np.all(np.diff(s) >= 0) for s in S))
    rho = [spearmanr(INV, s)[0] if np.ptp(s) > 0 else 0.0 for s in S]
    out[m] = {"n": len(keys), "monotonic": mono, "spearman_mean": float(np.nanmean(rho)),
              "pearson_mean_strength": float(pearsonr(INV, S.mean(0))[0]), "mean_strength": S.mean(0).tolist(),
              "ssim": Q.mean(0).tolist()}
    print(f"{m:11s} mono {mono}/{len(keys)}  spearman {np.nanmean(rho):.3f}  pearson(mean) {out[m]['pearson_mean_strength']:.3f}  "
          f"strength {np.round(S.mean(0), 3).tolist()}  ssim {np.round(Q.mean(0), 4).tolist()}")
(O / "modest_summary.json").write_text(json.dumps(out, indent=1))
