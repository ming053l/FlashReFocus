#!/usr/bin/env python
"""Restore the held-out model-selection sets with a Stage I checkpoint (Sec. 4.1): DPDD val_c (74 pairs) and LFDOF test
(50 scenes, the middle focal slice of each), at full resolution, used in neither training nor the reported results.
Outputs are named like the ground truth, for eval/score_folders.py. No refiner, detail weight 1."""
import argparse
from pathlib import Path

from flashrefocus import paths
from flashrefocus.deblurring import Deblurrer
from flashrefocus.image_io import read_image, save_image, to_signed, to_unit_range

DPDD_VAL = paths.DATA / "dd_dp_dataset_png/val_c"
LFDOF_TEST = paths.DATA / "LFDOF/test_data"


def heldout_pairs(name):
    if name == "dpdd_val":  # DPDD pairs are consecutive captures with different names: pair in sorted order
        return list(zip(sorted((DPDD_VAL / "source").glob("*.png")), sorted((DPDD_VAL / "target").glob("*.png"))))
    scenes = sorted(p for p in (LFDOF_TEST / "input").iterdir() if p.is_dir())
    middle = lambda files: files[len(files) // 2]  # middle focal slice (scenes have 3-15 slices)
    return [(middle(sorted(s.glob("*.png"))), LFDOF_TEST / "ground_truth" / f"{s.name}.png") for s in scenes]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--set", choices=["dpdd_val", "lfdof_test"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    deblurrer = Deblurrer.from_pretrained(a.checkpoint, refiner=None, detail_weight=1.0)
    a.out.mkdir(parents=True, exist_ok=True)
    pairs = heldout_pairs(a.set)
    for src, gt in pairs:
        save_image(to_unit_range(deblurrer(to_signed(read_image(src, "cuda", multiple=8)))), a.out / gt.name)
    print(f"{a.set}: {len(pairs)} images -> {a.out}")


if __name__ == "__main__":
    main()
