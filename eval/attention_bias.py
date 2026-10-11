#!/usr/bin/env python
"""Effect of the defocus bias on the renderer's attention (Appendix C.1, Fig. C.1).

For a query token q, the axial attention (rows, then columns) gives token (h', w') the effective weight
A_col[q, h'] * A_row[h', w'(q)]. Every defocus-bias attention block is re-evaluated on its own inputs with and without
the bias term gamma_b * alpha * B (same queries, keys and values), and the weights are averaged over heads.

  attention_bias.py maps  <image> <focal x> <focal y> <K> <out.npz> '[[x, y], ...]'   weights per block and query
  attention_bias.py stats [--root <data>/3CReal/validation]                              share on similar defocus

stats: 3CReal resized to a long side of 1024, rendered at the annotated focal point with K = 0.5, 16 random queries per
image; reports the share of each query's attention on tokens whose defocus M differs from the query's by < 0.1."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
paths.add_to_path()
import fast_refocus as FR  # noqa: E402
from method.blocks import DefocusBiasAttention  # noqa: E402


class AttentionProbe:
    """Renders with the fast renderer in fp32 and keeps the inputs (x, M, K) of every defocus-bias attention block."""

    def __init__(self):
        FR.setup_backend(False)
        self.renderer = FR.FastRenderer(amp_dtype=None)
        self.calls = []
        for m in self.renderer.model.modules():
            if isinstance(m, DefocusBiasAttention):
                m.register_forward_hook(lambda mod, i, o: self.calls.append((mod, i[0].detach(), i[1].detach(), i[2])))

    def render(self, image, focal_x, focal_y, strength):
        self.renderer.set_image(image)
        self.calls.clear()
        return self.renderer.render(focal_x, focal_y, strength)


@torch.no_grad()
def effective_attention(mod, x, defocus, strength, queries):
    """Per query (token row, column): effective weights (h, w) with and without the bias, averaged over heads."""
    b, h, w, _ = x.shape
    defocus = F.interpolate(defocus.to(x.dtype), size=(h, w), mode="bilinear", align_corners=False)
    q = mod.q_proj(x).view(b, h, w, mod.num_heads, mod.key_dim).permute(0, 3, 1, 2, 4)[0]          # heads, h, w, d
    k = (mod.k_proj(x) * mod.scaling).view(b, h, w, mod.num_heads, mod.key_dim).permute(0, 3, 1, 2, 4)[0]
    bias_rows, bias_cols = mod._defocus_bias(defocus)
    scale = (mod._gamma_b(x, strength) * mod.defocus_alpha.view(1, 1, mod.num_heads, 1, 1))[0, 0, :, 0]  # heads, 1
    out = []
    for qy, qx in queries:
        maps = {}
        for name, s in (("bias", 1.0), ("no_bias", 0.0)):
            a_row = torch.softmax(torch.einsum("chd,chwd->hcw", q[:, :, qx], k) + s * scale * bias_rows[0, :, :, qx], -1)
            a_col = torch.softmax(torch.einsum("cd,chd->ch", q[:, qy, qx], k[:, :, qx]) + s * scale * bias_cols[0, qx, :, qy],
                                  -1)
            maps[name] = (a_col.t()[:, :, None] * a_row).mean(1)                                       # h, w
        out.append(maps)
    return out, defocus[0, 0]


def to_tokens(xy, image_hw, token_hw):
    return min(token_hw[0] - 1, int(xy[1] * token_hw[0] / image_hw[0])), min(token_hw[1] - 1, int(xy[0] * token_hw[1] / image_hw[1]))


def cmd_maps(a):
    probe = AttentionProbe()
    probe.render(a.image, a.focal_x, a.focal_y, a.strength)
    image_hw = probe.renderer.src.shape[-2:]
    result = {}
    for block, (mod, x, defocus, strength) in enumerate(probe.calls):
        token_hw = x.shape[1:3]
        maps, m = effective_attention(mod, x, defocus, strength, [to_tokens(q, image_hw, token_hw) for q in a.queries])
        result["defocus"] = m.cpu().numpy()
        for qi, mp in enumerate(maps):
            for name, v in mp.items():
                result.setdefault(f"q{qi}_{name}", []).append(v.cpu().numpy())
    np.savez(a.out, **{k: np.asarray(v) for k, v in result.items()})


def cmd_stats(a):
    probe = AttentionProbe()
    ann = json.loads((a.root / "focus_point.json").read_text())["annotations"]
    rng = np.random.default_rng(0)
    share = {"bias": [], "no_bias": []}
    tmp = paths.WORK / "attention_bias_input.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    for key in sorted(ann, key=int):
        im = Image.open(a.root / "original" / f"{key}.jpg").convert("RGB")
        s = 1024 / max(im.size)
        w, h = int(im.size[0] * s) // 16 * 16, int(im.size[1] * s) // 16 * 16
        im.resize((w, h), Image.BICUBIC).save(tmp)
        probe.render(tmp, ann[key]["x"] * w / ann[key]["width"], ann[key]["y"] * h / ann[key]["height"], 0.5)
        th, tw = probe.calls[0][1].shape[1:3]
        queries = [(int(rng.integers(th)), int(rng.integers(tw))) for _ in range(16)]
        per_block = {"bias": [], "no_bias": []}
        for mod, x, defocus, strength in probe.calls:
            maps, m = effective_attention(mod, x, defocus, strength, queries)
            for mp, (qy, qx) in zip(maps, queries):
                similar = ((m - m[qy, qx]).abs() < 0.1).float()
                for name in per_block:
                    per_block[name].append(float((mp[name] * similar).sum()))
        for name in share:
            share[name] += list(np.asarray(per_block[name]).reshape(len(probe.calls), -1).mean(0))
    b, n = np.array(share["bias"]), np.array(share["no_bias"])
    print(f"{len(b)} queries: share of attention on similar defocus {n.mean():.3f} without the bias, {b.mean():.3f} with it")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("maps")
    m.add_argument("image"); m.add_argument("focal_x", type=float); m.add_argument("focal_y", type=float)
    m.add_argument("strength", type=float); m.add_argument("out"); m.add_argument("queries", type=json.loads)
    s = sub.add_parser("stats")
    s.add_argument("--root", type=Path, default=paths.DATA / "3CReal" / "validation")
    a = ap.parse_args()
    {"maps": cmd_maps, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
