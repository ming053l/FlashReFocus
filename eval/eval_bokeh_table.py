#!/usr/bin/env python
"""Score the saved outputs of every method in the bokeh-rendering tables (EBB! Val176, 3CReal, RealBokeh validation)
with one metric implementation: PSNR, SSIM, MS-SSIM, LPIPS (AlexNet and VGG) and DISTS from torchmetrics. Predictions are
matched to the ground truth by the leading integer of the file name and resized to the target size if needed.
No-reference metrics and CLIP-I are computed by score_folders.py."""

import argparse
import sys
import csv
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
paths.add_to_path()

# Ground truth under FLASHREFOCUS_DATA; each method's outputs under FLASHREFOCUS_WORK/bokeh_outputs/<method>/<dataset>/,
# matched to the ground truth by the leading integer of the file name.
GT = {
    "ebb": paths.DATA / "EBB_Val176" / "bokeh",
    "3creal": paths.DATA / "3CReal" / "validation" / "bokeh",
    "rbval": paths.DATA / "RealBokeh_val_f2.0" / "bokeh",          # RealBokeh validation, f/2.0 targets
}
METHODS = ["DeepLens", "MPIB", "BokehMe", "Dr.Bokeh", "Bokehlicious", "BokehDiff", "GenRefocus", "MagicBokeh", "FlashReFocus"]
OUTPUTS = paths.WORK / "bokeh_outputs"
PRED = {m: {ds: OUTPUTS / m / ds for ds in GT} for m in METHODS}
SUFFIXES = {".png", ".jpg", ".jpeg"}


def key_of(path):
    m = re.match(r"(\d+)", path.stem)
    return m.group(1) if m else None


def index(folder, recursive=False):
    files = folder.rglob("*") if recursive else folder.iterdir()
    out = {}
    for p in files:
        if p.is_file() and p.suffix.lower() in SUFFIXES:
            k = key_of(p)
            if k is not None:
                out.setdefault(k, []).append(p)
    return out


def load(path, device):
    with Image.open(path) as im:
        return (pil_to_tensor(im.convert("RGB")).float() / 255.0).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(PRED))
    ap.add_argument("--datasets", nargs="*", default=["ebb", "3creal"])
    ap.add_argument("--out", type=Path, default=paths.WORK / "bokeh_results")
    ap.add_argument("--summary_name", default="summary.json")
    ap.add_argument("--inspect", action="store_true", help="only report matching, no metrics")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    dev = torch.device("cuda")

    if not args.inspect:
        from torchmetrics.image import (MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio,
                                        StructuralSimilarityIndexMeasure)
        from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        M = {
            "psnr": PeakSignalNoiseRatio(data_range=1.0).to(dev),
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(dev),
            "lpips": LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev).eval(),
            "dists": DeepImageStructureAndTextureSimilarity().to(dev).eval(),
            "ms_ssim": MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(dev),
            "lpips_vgg": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev).eval(),
        }

    summary = {}
    for ds in args.datasets:
        gt = index(GT[ds])
        for meth in args.methods:
            if ds not in PRED[meth]:
                continue
            folder = PRED[meth][ds]
            if not folder.exists():
                print(f"[{meth}/{ds}] MISSING {folder}")
                continue
            pred = index(folder, recursive=(meth == "Bokehlicious"))
            keys = sorted(set(gt) & set(pred), key=int)
            multi = sum(len(pred[k]) > 1 for k in keys)
            print(f"[{meth}/{ds}] gt={len(gt)} pred={len(pred)} matched={len(keys)} multi-candidates={multi}")
            if args.inspect:
                for k in keys[:2]:
                    print("   ", k, [p.name for p in pred[k]][:4])
                continue
            rows, resized = [], 0
            with torch.no_grad():
                for k in keys:
                    p = load(sorted(pred[k])[0], dev)
                    g = load(gt[k][0], dev)
                    if p.shape != g.shape:
                        p = F.interpolate(p, size=g.shape[-2:], mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
                        resized += 1
                    r = {"key": k}
                    for name, fn in M.items():
                        r[name] = float(fn(p, g))
                        fn.reset()
                    rows.append(r)
            with open(args.out / f"{meth}_{ds}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
            mean = {n: sum(r[n] for r in rows) / len(rows) for n in M}
            mean.update(n=len(rows), resized=resized)
            summary[f"{meth}/{ds}"] = mean
            print("   ", {n: round(v, 4) for n, v in mean.items()})
    if not args.inspect:
        (args.out / args.summary_name).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
