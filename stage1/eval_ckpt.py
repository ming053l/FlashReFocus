#!/usr/bin/env python
"""Restore the held-out model-selection sets with a Stage I checkpoint: DPDD val_c (74 pairs) and LFDOF test (50 scenes,
one focal slice each), both at full resolution and used in neither training nor the reported results. Outputs are named
like the ground truth, for eval/score_folders.py and sharpness.py. Inference as deblur_no_refiner.py."""
import os
import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

# datasets under FLASHREFOCUS_DATA, outputs under FLASHREFOCUS_WORK (defaults <repo>/data, <repo>/work; see paths.py)
_REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FLASHREFOCUS_DATA", _REPO / "data"))
WORK_ROOT = Path(os.environ.get("FLASHREFOCUS_WORK", _REPO / "work"))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repository root
import paths  # noqa: E402
paths.add_to_path()
from deblur_no_refiner import Stage1Model  # noqa: E402

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
    import argparse as _ap
    args = _ap.Namespace(pretrained_model_path=str(paths.SD21), pretrained_path=a.ckpt,
                         lambda_pix=1.0, lambda_sem=1.0, mixed_precision="fp16", default=True,
                         vae_decoder_tiled_size=100000, vae_encoder_tiled_size=100000,
                         latent_tiled_size=100000, latent_tiled_overlap=0, seed=42)
    model = Stage1Model(args)
    model.set_eval()
    a.out.mkdir(parents=True, exist_ok=True)
    for src, gt in heldout_pairs(a.set):
        img = Image.open(src).convert("RGB")
        w, h = img.size
        img = img.crop((0, 0, w - w % 8, h - h % 8))
        out = model(True, TF.to_tensor(img).unsqueeze(0).cuda() * 2 - 1, prompt="")
        TF.to_pil_image((out[0].float() * 0.5 + 0.5).clamp(0, 1).cpu()).save(a.out / gt.name)
    print(f"{a.set}: {len(heldout_pairs(a.set))} images -> {a.out}")


if __name__ == "__main__":
    main()
