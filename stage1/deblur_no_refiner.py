#!/usr/bin/env python
"""Stage I inference without the refiner (SD2.1 VAE decoder) on the DPDD / RealDOF test pairs, with options for latent
tiling, input rescaling, RAM++ prompts and AdaIN colour alignment. Pairs come from pairs/<set>_test.json (paths
relative to the data root); outputs are written as {index:04d}.png. The model itself is in model.py."""
import argparse
import json
import time
from pathlib import Path
import sys

import torch
import torchvision.transforms.functional as TF
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repository root
import paths  # noqa: E402
paths.add_to_path()
from model import Stage1Model, load_stage1  # noqa: E402,F401  (Stage1Model: kept importable from here)
from src.my_utils.wavelet_color_fix import adain_color_fix  # noqa: E402

PAIRS = {"dpdd": HERE / "pairs/dpdd_test.json", "realdof": HERE / "pairs/realdof_test.json"}   # paths relative to paths.DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(PAIRS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--align", choices=["nofix", "adain"], default="nofix")
    ap.add_argument("--lambda_pix", type=float, default=1.0)
    ap.add_argument("--lambda_sem", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--latent_tile", type=int, default=100000, help="latent tile size; default = whole image")
    ap.add_argument("--latent_overlap", type=int, default=0, help="latent tile overlap")
    ap.add_argument("--scale", type=float, default=1.0, help="deblur at this input scale, then resize back")
    ap.add_argument("--ram", action="store_true", help="prompt = RAM++ tags of the blurred input")
    ap.add_argument("--ckpt", default=str(paths.STAGE1), help="Stage I checkpoint (.pkl)")
    a = ap.parse_args()
    model = load_stage1(a.ckpt, omega=a.lambda_sem, lambda_pix=a.lambda_pix, latent_tile=a.latent_tile,
                        latent_overlap=a.latent_overlap)
    tagger = None
    if a.ram:
        from ram_plus_tagger import RamPlusTagger
        tagger = RamPlusTagger(checkpoint_path=str(paths.RAM_PLUS),
                               image_size=384, vit="swin_l", fallback_prompt="a sharp, clear image", prompt_threshold=0.7,
                               max_tags=50).cuda().eval()
    samples = json.load(open(PAIRS[a.dataset]))[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    times = []
    for s in samples:
        img = Image.open(paths.DATA / s["source"]).convert("RGB")
        w, h = img.size
        img = img.crop((0, 0, w - w % 8, h - h % 8))
        full = img.size
        if a.scale != 1.0:
            img = img.resize((round(full[0] * a.scale / 8) * 8, round(full[1] * a.scale / 8) * 8), Image.BICUBIC)
        c_t = TF.to_tensor(img).unsqueeze(0).cuda() * 2 - 1
        torch.cuda.synchronize(); t0 = time.perf_counter()
        prompt = tagger.predict_batch(c_t.float() * 0.5 + 0.5).prompt_texts[0] if tagger else ""
        out = model(model.single_pass, c_t, prompt=prompt)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        pil = TF.to_pil_image((out[0].float() * 0.5 + 0.5).clamp(0, 1).cpu())
        if pil.size != full:
            pil = pil.resize(full, Image.BICUBIC)
        if a.align == "adain":
            pil = adain_color_fix(target=pil, source=img)
        pil.save(a.out / f"{s['index']:04d}.png")
    warm = times[3:] or times
    print(f"{a.dataset} {len(samples)} images, mean time {sum(warm) / len(warm):.3f}s")


if __name__ == "__main__":
    main()
