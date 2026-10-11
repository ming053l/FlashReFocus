#!/usr/bin/env python
"""Strength controllability on MODEST aperture series: for each method, the per-image best strength (in the method's
own unit) found against the f/16, 9, 5 and 2.8 targets is compared with 1/N: the share of series whose best strength
never decreases as the aperture opens, the mean Spearman rank correlation per series, and the Pearson correlation of the
mean strength with 1/N; plus the mean SSIM per aperture. Reads <work>/rbval_out/modest_*; writes modest_summary.json."""
import argparse
import csv
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

from flashrefocus import paths

RESULTS = paths.WORK / "rbval_out"   # strength-search outputs of every method on the MODEST series
APERTURES = ["F16.0", "F9.0", "F5.0", "F2.8"]
INVERSE_F_NUMBER = np.array([1 / float(a[1:]) for a in APERTURES])
# method -> (results file per aperture, key column, strength column, SSIM column)
RESULT_FILES = {"FlashReFocus": ("modest_ours_bf16_{}/results.csv", "key", "strength", "ssim"),
                "BokehMe": ("modest_bokehme_{}/results.csv", "scene_id", "best_k", "ssim"),
                "MagicBokeh": ("modest_magicbokeh_{}/metrics_per_image_shard0.csv", "key", "k", "ssim"),
                "DeepLens": ("modest_deeplens_{}/metrics.csv", "image", "aperture", "ssim")}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    summary = {}
    for method, (pattern, key_col, strength_col, ssim_col) in RESULT_FILES.items():
        table = {}
        for i, aperture in enumerate(APERTURES):
            for row in csv.DictReader(open(RESULTS / pattern.format(aperture))):
                table.setdefault(row[key_col].split(".")[0], [None] * 4)[i] = (float(row[strength_col]),
                                                                             float(row[ssim_col]))
        keys = sorted(k for k in table if all(table[k]))
        strength = np.array([[table[k][i][0] for i in range(4)] for k in keys])
        ssim = np.array([[table[k][i][1] for i in range(4)] for k in keys])
        monotonic = int(sum(np.all(np.diff(s) >= 0) for s in strength))
        rho = [spearmanr(INVERSE_F_NUMBER, s)[0] if np.ptp(s) > 0 else 0.0 for s in strength]
        summary[method] = {"n": len(keys), "monotonic": monotonic, "spearman_mean": float(np.nanmean(rho)),
                           "pearson_mean_strength": float(pearsonr(INVERSE_F_NUMBER, strength.mean(0))[0]),
                           "mean_strength": strength.mean(0).tolist(), "ssim": ssim.mean(0).tolist()}
        print(f"{method:11s} monotonic {monotonic}/{len(keys)}  spearman {np.nanmean(rho):.3f}  "
              f"pearson(mean) {summary[method]['pearson_mean_strength']:.3f}  "
              f"strength {np.round(strength.mean(0), 3).tolist()}  ssim {np.round(ssim.mean(0), 4).tolist()}")
    (RESULTS / "modest_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
