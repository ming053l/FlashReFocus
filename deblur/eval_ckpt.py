#!/usr/bin/env python
"""Restore the held-out model-selection sets with a Stage I checkpoint: DPDD val_c (74 pairs) and LFDOF test (50 scenes,
one focal slice each), both at full resolution and used in neither training nor the reported results. Outputs are named
like the ground truth, for eval/score_folders.py. Inference as deblur.py without the refiner."""
import os
import argparse
import sys
from pathlib import Path

# datasets under FLASHREFOCUS_DATA, outputs under FLASHREFOCUS_WORK (defaults <repo>/data, <repo>/work; see paths.py)
_REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FLASHREFOCUS_DATA", _REPO / "data"))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repository root
import paths  # noqa: E402
paths.add_to_path()
from model import Deblurrer, load_stage1, read_image, save_image  # noqa: E402

DPDD_VAL = paths.DATA / "dd_dp_dataset_png/val_c"
LFDOF_TEST = (DATA_ROOT / "LFDOF/test_data")


def heldout_pairs(name):
    if name == "dpdd_val":  # DPDD pairs are consecutive captures with different names: pair in sorted order
        return list(zip(sorted((DPDD_VAL / "source").glob("*.png")), sorted((DPDD_VAL / "target").glob("*.png"))))
    scenes = sorted(p for p in (LFDOF_TEST / "input").iterdir() if p.is_dir())
    middle = lambda files: files[len(files) // 2]  # middle focal slice (scenes have 3-15 slices)
    return [(middle(sorted(s.glob("*.png"))), LFDOF_TEST / "ground_truth" / f"{s.name}.png") for s in scenes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", choices=["dpdd_val", "lfdof_test"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    deblur = Deblurrer(load_stage1(a.ckpt))
    a.out.mkdir(parents=True, exist_ok=True)
    for src, gt in heldout_pairs(a.set):
        save_image(deblur(read_image(src)), a.out / gt.name)
    print(f"{a.set}: {len(heldout_pairs(a.set))} images -> {a.out}")


if __name__ == "__main__":
    main()
