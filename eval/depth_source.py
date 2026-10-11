#!/usr/bin/env python
"""Depth source study (Appendix G, Table G.4): the same renderer and strength search with depth from DA-V2 S/B/L or
from Depth Pro, without retraining. Benchmarks in the original/, bokeh/, focus_point.json layout.

  depth_source.py dump-depthpro <root> <out_dir>      Depth Pro inverse depth per input (run where depth_pro is installed)
  depth_source.py eval <vits|vitb|vitl|depthpro> <root> <out_dir> [--depthpro-dir DIR] [--da2-checkpoint PTH]
DA-V2 S/B checkpoints: huggingface.co/depth-anything/Depth-Anything-V2-{Small,Base}."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from flashrefocus import Refocuser
from flashrefocus.image_io import read_image, save_image
from flashrefocus.rendering.depth import DepthEstimator, disparity_to_depth, load_depth_anything_v2
from flashrefocus.rendering.depth_edits import depth_at
from bokeh_strength_search import annotated_focal_point, search_strength


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
    from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    (out / "images").mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    rf = Refocuser.from_pretrained(deblurring=False)
    if source in ("vitb", "vits"):
        rf.depth_estimator = DepthEstimator(load_depth_anything_v2(source, da2_checkpoint)).cuda()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda().eval()
    dists = DeepImageStructureAndTextureSimilarity().cuda().eval()
    annotations = json.loads((root / "focus_point.json").read_text())["annotations"]
    rows = []
    for key in sorted(annotations, key=int):
        scene = rf.scene_from_file(root / "original" / f"{key}.jpg")
        depth = scene.depth
        if source == "depthpro":   # D = 1 - minmax(inverse depth), as for DA-V2
            depth = disparity_to_depth(torch.from_numpy(np.load(depthpro_dir / f"{key}.npy")).cuda()[None])
        x, y = annotated_focal_point(annotations[key], *scene.image.shape[-2:])
        focal_depth = float(depth_at(depth, x, y))
        target = read_image(root / "bokeh" / f"{key}.jpg", "cuda")
        k, _, bokeh, _ = search_strength(rf, scene, focal_depth, target, depth=depth)
        if bokeh.shape != target.shape:
            bokeh = torch.nn.functional.interpolate(bokeh, target.shape[-2:], mode="bilinear", align_corners=False)
        with torch.no_grad():
            rows.append({"key": key, "k": k, "psnr": float(psnr(bokeh, target, data_range=1.0)),
                         "ssim": float(ssim(bokeh, target, data_range=1.0)), "lpips": float(lpips(bokeh, target)),
                         "dists": float(dists(bokeh, target))})
        save_image(bokeh, out / "images" / f"{key}.png")
    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
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
