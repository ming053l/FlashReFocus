#!/usr/bin/env python
"""Stage I inference on a list of pairs: the one-step diffusion model restores the latent (omega = --lambda_sem; 1.25 for
the released model) and the decoder refiner decodes it, guided by the blurred input (without --refiner: the SD2.1 VAE
decoder).

Pairs come from pairs/<set>_test.json (paths relative to --data_root); outputs are written as {index:04d}.png.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repository root
import paths  # noqa: E402
paths.add_to_path()
from model import Deblurrer, load_refiner, load_stage1, read_image, save_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--data_root", default=str(paths.DATA))
    ap.add_argument("--ckpt", default=str(paths.STAGE1), help="Stage I checkpoint (.pkl)")
    ap.add_argument("--refiner", default=None, help="refiner.pt; omit to decode with the SD2.1 VAE decoder")
    ap.add_argument("--sd21", default=str(paths.SD21))
    ap.add_argument("--sd15", default=str(paths.SD15))
    ap.add_argument("--lambda_pix", type=float, default=1.0)
    ap.add_argument("--lambda_sem", type=float, default=1.0, help="detail weight omega; != 1 runs two U-Net passes")
    ap.add_argument("--sem_scale", type=float, default=1.0,
                    help="scale the merged detail-LoRA update in weight space (one U-Net pass; alternative to --lambda_sem)")
    ap.add_argument("--refiner_tile", type=int, default=0, help="pixel tile for the refiner (0 = whole image)")
    ap.add_argument("--refiner_overlap", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    model = load_stage1(a.ckpt, a.sd21, omega=a.lambda_sem, lambda_pix=a.lambda_pix)
    if a.sem_scale != 1.0:
        assert model.single_pass, "--sem_scale replaces --lambda_sem; use one of them"
        # W = W_pix + delta_sem (merged); ori_unet_weight = W_pix  ->  W_pix + s * delta_sem
        for n, p in model.unet.named_parameters():
            o = model.ori_unet_weight[n]
            p.data = o + a.sem_scale * (p.data - o)
    deblur = Deblurrer(model, load_refiner(a.refiner, a.sd15) if a.refiner else None)

    a.out.mkdir(parents=True, exist_ok=True)
    times = []
    for s in json.load(open(a.pairs)):
        x = read_image(Path(a.data_root) / s["source"])
        torch.cuda.synchronize(); t0 = time.perf_counter()
        y = deblur(x, a.refiner_tile, a.refiner_overlap)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        save_image(y.clamp(-1, 1), a.out / f"{s['index']:04d}.png")
    warm = times[3:] or times
    print(f"{a.pairs.stem}: {len(times)} images, mean time {sum(warm) / len(warm):.3f}s"
          + (" (excluding the first 3)" if len(times) > 3 else ""))


if __name__ == "__main__":
    main()
