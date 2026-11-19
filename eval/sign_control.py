#!/usr/bin/env python
"""Where the side of the focal plane comes from (appendix on the defocus-bias attention). The defocus bias reads the magnitude M = |M^s|, so a
token in front of the focal plane and one behind it with the same blur get the same bias; the side is given only by
the signed condition M^s (and M^fg) in M_C.

  sign_control.py attention [--root <data>/3CReal/validation]
      Protocol of attention_bias.py stats (long side 1024, annotated focal point, K = 0.5, 16 random queries per
      image), restricted to queries with M >= 0.1. Among the tokens of similar blur (|M_j - M_q| < 0.1), reports the
      share of attention on the same side of the focal plane as the query and on the opposite side, with and
      without the bias.
  sign_control.py render --variant {signed,unsigned,flipped} --out <dir> [--root ...]
      Bokeh-table protocol (bokeh_strength_search.py: strength k/100 with the highest SSIM, native resolution), with
      M^s in M_C kept (signed), replaced by |M^s| (unsigned; M^fg then marks only the focal plane) or negated
      (flipped). The attention input M is the same in all three. Writes <out>/results.csv with PSNR, SSIM, LPIPS and
      the share of pixels in front of the focal plane by more than 0.05."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from flashrefocus import paths
from flashrefocus.image_io import read_image, save_image
from flashrefocus.rendering import renderer as renderer_module

from attention_bias import AttentionProbe, effective_attention
from bokeh_strength_search import annotated_focal_point, search_strength

SIGN_VARIANTS = {"signed": lambda s: s, "unsigned": torch.abs, "flipped": torch.neg}


def use_sign_variant(name):
    """Replace M^s in M_C by SIGN_VARIANTS[name](M^s); M = |M^s| for the attention is unchanged."""
    original = renderer_module.condition_maps

    def condition_maps(depth, focal_depth, strength, size):
        fd = focal_depth.to(depth).view(-1, 1, 1, 1)
        _, cond = original(SIGN_VARIANTS[name](depth - fd) + fd, focal_depth, strength, size)
        return depth - fd, cond

    renderer_module.condition_maps = condition_maps


def cmd_attention(a):
    probe = AttentionProbe()
    ann = json.loads((a.root / "focus_point.json").read_text())["annotations"]
    rng = np.random.default_rng(0)
    tmp = paths.WORK / "attention_bias_input.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    rows = []   # per query: shares (same side, opposite side) with and without the bias, and the token fractions
    for key in sorted(ann, key=int):
        im = Image.open(a.root / "original" / f"{key}.jpg").convert("RGB")
        s = 1024 / max(im.size)
        w, h = int(im.size[0] * s) // 16 * 16, int(im.size[1] * s) // 16 * 16
        im.resize((w, h), Image.BICUBIC).save(tmp)
        fx, fy = ann[key]["x"] * w / ann[key]["width"], ann[key]["y"] * h / ann[key]["height"]
        probe.render(tmp, fx, fy, 0.5)
        scene = probe.refocuser.scene_from_file(tmp)
        signed = scene.depth[None] - scene.focal_depth_at(fx, fy)
        th, tw = probe.calls[0][1].shape[1:3]
        sign = torch.sign(F.interpolate(signed.float(), size=(th, tw), mode="bilinear", align_corners=False)[0, 0])
        queries = [(int(rng.integers(th)), int(rng.integers(tw))) for _ in range(16)]
        per_block = []
        for mod, x, defocus, strength in probe.calls:
            maps, m = effective_attention(mod, x, defocus, strength, queries)
            for mp, (qy, qx) in zip(maps, queries):
                similar = (m - m[qy, qx]).abs() < 0.1
                same, opposite = (similar & (sign == sign[qy, qx])).float(), (similar & (sign == -sign[qy, qx])).float()
                per_block.append([float(m[qy, qx])] + [float((mp[n] * sel).sum()) for n in ("bias", "no_bias")
                                                       for sel in (same, opposite)] + [float(opposite.mean())])
        rows += list(np.asarray(per_block).reshape(len(probe.calls), len(queries), -1).mean(0))
    r = np.asarray(rows)
    r = r[r[:, 0] >= 0.1]
    r = r[r[:, 5] > 0]   # queries with similar-blur tokens on the other side of the focal plane
    print(f"{len(r)} queries with M >= 0.1 and similar-blur tokens on both sides")
    print(f"  same side:     {r[:, 3].mean():.3f} without the bias -> {r[:, 1].mean():.3f} with it")
    print(f"  opposite side: {r[:, 4].mean():.3f} without the bias -> {r[:, 2].mean():.3f} with it")


def cmd_render(a):
    from flashrefocus import Refocuser
    use_sign_variant(a.variant)
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    rf = Refocuser.from_pretrained(deblurring=False)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda().eval()
    ann = json.loads((a.root / "focus_point.json").read_text())["annotations"]
    rows = []
    for key in sorted(ann, key=int):
        src = a.root / "original" / f"{key}.jpg"
        if not src.exists():
            continue
        scene = rf.scene_from_file(src)
        focal_depth = scene.focal_depth_at(*annotated_focal_point(ann[key], *scene.image.shape[-2:]))
        target = read_image(a.root / "bokeh" / f"{key}.jpg", "cuda")
        k, _, bokeh, _ = search_strength(rf, scene, focal_depth, target)
        save_image(bokeh, a.out / "images" / f"{key}.png")
        with torch.no_grad():
            rows.append({"key": key, "best_k": k,
                         "psnr": float(peak_signal_noise_ratio(bokeh, target, data_range=1.0)),
                         "ssim": float(structural_similarity_index_measure(bokeh, target, data_range=1.0)),
                         "lpips": float(lpips(bokeh, target)),
                         "front_share": float((scene.depth - focal_depth < -0.05).float().mean())})
        print(rows[-1], flush=True)
    with open(a.out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(a.variant, {n: round(sum(r[n] for r in rows) / len(rows), 4) for n in ("psnr", "ssim", "lpips")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("attention", "render"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=paths.DATA / "3CReal" / "validation")
    sub.choices["render"].add_argument("--variant", choices=list(SIGN_VARIANTS), required=True)
    sub.choices["render"].add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"attention": cmd_attention, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    main()
