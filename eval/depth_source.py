#!/usr/bin/env python
"""Depth source study (Appendix G, Table G.4): the same renderer and strength search with depth from DA-V2 S/B/L or
from Depth Pro, without retraining. Benchmarks in the original/, bokeh/, focus_point.json layout.

  depth_source.py dump-depthpro <root> <out_dir>      Depth Pro inverse depth per input (run where depth_pro is installed)
  depth_source.py eval <vits|vitb|vitl|depthpro> <root> <out_dir> [--depthpro-dir DIR] [--da2-checkpoint PTH]
DA-V2 S/B checkpoints: huggingface.co/depth-anything/Depth-Anything-V2-{Small,Base}."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402


def dump_depthpro(root, out):
    import cv2
    import depth_pro
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    model, transform = depth_pro.create_model_and_transforms(device=torch.device("cuda"))
    model.eval()
    for key in sorted(json.loads((root / "focus_point.json").read_text())["annotations"], key=int):
        im = Image.open(root / "original" / f"{key}.jpg").convert("RGB")
        w, h = im.size
        with torch.no_grad():
            z = model.infer(transform(np.asarray(im)).cuda(), f_px=None)["depth"].cpu().numpy().squeeze()
        z = cv2.resize(z.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        np.save(out / f"{key}.npy", (1.0 / np.where(z > 0, z, np.finfo(np.float32).max)).astype(np.float32))


def evaluate(source, root, out, depthpro_dir, da2_checkpoint):
    paths.add_to_path()
    import fast_refocus as FR
    from infer_single_focus import load_da2_model
    from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    (out / "images").mkdir(parents=True, exist_ok=True)
    FR.setup_backend(False)
    rd = FR.FastRenderer()
    if source in ("vitb", "vits"):
        rd.da2 = load_da2_model(source, da2_checkpoint, torch.device("cuda"))
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda().eval()
    dists = DeepImageStructureAndTextureSimilarity().cuda().eval()
    ann = json.loads((root / "focus_point.json").read_text())["annotations"]
    rows = []
    for key in sorted(ann, key=int):
        rd.set_image(root / "original" / f"{key}.jpg")
        h, w = rd.src.shape[-2:]
        depth = None
        if source == "depthpro":   # D = 1 - min-max(inverse depth), as for DA-V2
            disp = torch.from_numpy(np.load(depthpro_dir / f"{key}.npy")).cuda()[None]
            depth = (1.0 - (disp - disp.min()) / (disp.max() - disp.min())).clamp(0, 1)
        fx, fy = ann[key]["x"] * w / ann[key]["width"], ann[key]["y"] * h / ann[key]["height"]
        gt = FR.read_rgb(root / "bokeh" / f"{key}.jpg", "cuda")
        cache = {}

        def score(k):
            if k not in cache:
                with torch.no_grad():
                    p = rd.render(fx, fy, k / 100.0, None, depth=depth).float().clamp(0, 1)
                    p = (p * 255.0).to(torch.uint8).float() / 255.0
                    cache[k] = (float(ssim(p, gt, data_range=1.0)), p)
            return cache[k][0]

        lo, hi = 1, 100
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (mid + 1, hi) if score(mid) < score(mid + 1) else (lo, mid)
        score(lo)
        p = cache[lo][1]
        if p.shape != gt.shape:
            p = torch.nn.functional.interpolate(p, gt.shape[-2:], mode="bilinear", align_corners=False)
        with torch.no_grad():
            rows.append({"key": key, "k": lo, "psnr": float(psnr(p, gt, data_range=1.0)),
                         "ssim": float(ssim(p, gt, data_range=1.0)), "lpips": float(lpips(p, gt)),
                         "dists": float(dists(p, gt))})
        FR.write_rgb(p, out / "images" / f"{key}.png")
    with open(out / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(source, {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("psnr", "ssim", "lpips", "dists")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump-depthpro")
    d.add_argument("root", type=Path); d.add_argument("out", type=Path)
    e = sub.add_parser("eval")
    e.add_argument("source", choices=["vits", "vitb", "vitl", "depthpro"])
    e.add_argument("root", type=Path); e.add_argument("out", type=Path)
    e.add_argument("--depthpro-dir", type=Path)
    e.add_argument("--da2-checkpoint", type=Path)
    a = ap.parse_args()
    if a.cmd == "dump-depthpro":
        dump_depthpro(a.root, a.out)
    else:
        evaluate(a.source, a.root, a.out, a.depthpro_dir, a.da2_checkpoint)


if __name__ == "__main__":
    main()
