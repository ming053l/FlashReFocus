#!/usr/bin/env python
"""Score prediction folders against a ground-truth folder.

Metrics follow GenRefocus (arXiv:2512.16923, Tables 3-4): LPIPS, DISTS, CLIP-I
(cosine similarity of OpenAI CLIP ViT-L/14 image embeddings of prediction and target),
CLIP-IQA and MUSIQ, plus PSNR and SSIM. Predictions are matched to ground truth by
the leading integer / stem of the file name and resized to the target size if needed.

usage: score_folders.py --gt DIR --pred NAME=DIR [--pred NAME=DIR ...] --out results.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SUF = {".png", ".jpg", ".jpeg"}


def key(p):
    m = re.match(r"(\d+)", p.stem)
    return str(int(m.group(1))) if m else p.stem


def index(d, recursive=False):
    it = Path(d).rglob("*") if recursive else Path(d).iterdir()
    out = {}
    for p in it:
        if p.is_file() and p.suffix.lower() in SUF:
            out.setdefault(key(p), p)
    return out


def load(p, dev):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", default="psnr,ssim,lpips,dists,clip_i,clipiqa,musiq")
    ap.add_argument("--recursive", default="", help="comma list of NAMEs whose folders are searched recursively")
    ap.add_argument("--keys", default=None, help="optional file with one key per line to restrict evaluation")
    a = ap.parse_args()
    dev = torch.device("cuda")
    want = a.metrics.split(",")

    import pyiqa
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    M = {}
    if "psnr" in want: M["psnr"] = PeakSignalNoiseRatio(data_range=1.0).to(dev)
    if "ssim" in want: M["ssim"] = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    if "lpips" in want: M["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev).eval()
    if "dists" in want: M["dists"] = DeepImageStructureAndTextureSimilarity().to(dev).eval()
    NR = {n: pyiqa.create_metric(n, device=dev) for n in ("clipiqa", "musiq", "maniqa") if n in want}
    clip = None
    if "clip_i" in want:
        import clip as openai_clip
        clip, preprocess = openai_clip.load("ViT-L/14", device=dev)
        clip.eval()

        def emb(t):
            img = Image.fromarray((t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            return F.normalize(clip.encode_image(preprocess(img)[None].to(dev)).float(), dim=-1)

    gt = index(a.gt)
    only = set(Path(a.keys).read_text().split()) if a.keys else None
    rec = set(filter(None, a.recursive.split(",")))
    res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for spec in a.pred:
        name, d = spec.split("=", 1)
        pred = index(d, recursive=name in rec)
        keys = sorted((set(gt) & set(pred)) - (set() if only is None else set(gt) - only), key=lambda s: (len(s), s))
        rows = []
        with torch.no_grad():
            for k in keys:
                p, g = load(pred[k], dev), load(gt[k], dev)
                if p.shape != g.shape:
                    p = F.interpolate(p, size=g.shape[-2:], mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
                r = {}
                for n, fn in M.items():
                    r[n] = float(fn(p, g)); fn.reset()
                for n, fn in NR.items():
                    r[n] = float(fn(p))
                if clip is not None:
                    r["clip_i"] = float((emb(p) * emb(g)).sum())
                rows.append(r)
        mean = {n: float(np.mean([r[n] for r in rows])) for n in rows[0]} if rows else {}
        mean["n"] = len(rows)
        res[name] = {**res.get(name, {}), **mean}
        print(name, {n: round(v, 4) for n, v in mean.items()}, flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
