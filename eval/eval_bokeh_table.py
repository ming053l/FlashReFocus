#!/usr/bin/env python
"""Re-score every method in the paper's zero-shot bokeh table (EBB! Val176, 3CReal)
from saved output images, with one metric implementation for all methods.

Existing columns (PSNR, SSIM, LPIPS-Alex, DISTS) use the same torchmetrics settings
as SOTA/Genfocus/infer_bokeh_folder_metrics.py so they can be checked against the
table; MS-SSIM and LPIPS-VGG are the added full-reference metrics.
Predictions are matched to ground truth by the leading integer in the file name.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

ROOT = Path(__file__).resolve().parent
SOTA = Path("/ssd1/chingheng/Defocus/SOTA")
GT = {
    "ebb": ROOT / "gt_ebb176" / "bokeh",
    "3creal": Path("/ssd1/chingheng/Deblurring-Dataset/3CReal/validation/bokeh"),
    "rbval": Path("/ssd1/chingheng/Defocus/refocus_bench/realbokeh_val/bokeh"),     # RealBokeh validation, f/2.0 targets
}
RB = Path("/ssd1/chingheng/Defocus/refocus_bench/rbval_out")
PRED = {
    "DeepLens": {"ebb": ROOT / "deeplens/ebb/dof", "3creal": SOTA / "deeplens_eval/data/res_3creal_validation/dof"},
    "MPIB": {"ebb": ROOT / "mpib/ebb", "3creal": ROOT / "mpib/3creal"},
    "BokehMe": {"ebb": ROOT / "bokehme/ebb/images", "3creal": ROOT / "bokehme/3creal/images"},
    "Dr.Bokeh": {"ebb": ROOT / "drbokeh/ebb/batch_eval", "3creal": ROOT / "drbokeh/3creal"},
    "Bokehlicious": {"ebb": ROOT / "bokehlicious/ebb/images/EBB400", "3creal": ROOT / "bokehlicious/3creal"},
    "BokehDiff": {"ebb": SOTA / "bokehdiff/bokehdiff_ebb_outputs_correct/validation/images",
                  "3creal": SOTA / "bokehdiff/bokehdiff_3creal_outputs/validation/images"},
    "GenRefocus": {"ebb": SOTA / "Genfocus/outputs/ebb_bokeh_eval/images", "3creal": SOTA / "Genfocus/outputs/bokeh_3creal/images"},
    "DeReFocus-B3": {"ebb": ROOT / "ours/ebb_B3"},
    "DeReFocus": {"ebb": ROOT / "ours/ebb/images", "3creal": ROOT / "ours/3creal/images"},
    "MagicBokeh": {"ebb": SOTA / "MagicBokeh/outputs/ebb_bokeh_eval/images", "3creal": SOTA / "MagicBokeh/outputs/bokeh_3creal/images"},
    "Abl-A0": {"3creal": ROOT / "ablation3c/A0_3creal/images"},
    "Abl-A1": {"3creal": ROOT / "ablation3c/A1_3creal/images"},
    "Abl-A2": {"3creal": ROOT / "ablation3c/A2_3creal/images"},
    "Abl-B2": {"3creal": ROOT / "ablation3c/B2_3creal/images"},
    "Abl-B4": {"3creal": ROOT / "ablation3c/B4_3creal/images"},
}
for _m, _d in {"DeepLens": "deeplens_rbval/dof", "BokehMe": "bokehme_rbval/images", "Bokehlicious": "bokehlicious/images",
               "BokehDiff": "bokehdiff_rbval/validation/images", "GenRefocus": "genrefocus/images", "MagicBokeh": "magicbokeh/images",
               "DeReFocus": "ours/images", "DeReFocus-bf16": "ours_bf16/images"}.items():
    PRED.setdefault(_m, {})["rbval"] = RB / _d
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
    ap.add_argument("--out", type=Path, default=ROOT / "results")
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
