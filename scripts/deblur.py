#!/usr/bin/env python
"""Stage I on a list of pairs (annotations/deblurring_pairs/<set>_test.json, paths relative to --data-root): the
one-step diffusion model restores the latent with detail weight omega (1.25 for the released model) and the decoder
refiner decodes it (--no-refiner: the SD2.1 VAE decoder). Outputs are written as {index:04d}.png."""
import argparse
import json
import time
from pathlib import Path

import torch

from flashrefocus import paths
from flashrefocus.deblurring import Deblurrer
from flashrefocus.image_io import read_image, save_image, to_signed, to_unit_range


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=paths.DATA)
    ap.add_argument("--checkpoint", type=Path, default=paths.STAGE1, help="Stage I checkpoint (.pkl)")
    ap.add_argument("--refiner", type=Path, default=paths.REFINER)
    ap.add_argument("--no-refiner", action="store_true")
    ap.add_argument("--detail-weight", type=float, default=1.25, help="omega; != 1 runs two U-Net passes")
    ap.add_argument("--refiner-tile", type=int, default=0, help="pixel tile for the refiner (0 = whole image)")
    ap.add_argument("--refiner-overlap", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    deblurrer = Deblurrer.from_pretrained(a.checkpoint, None if a.no_refiner else a.refiner, a.detail_weight)
    a.out.mkdir(parents=True, exist_ok=True)
    times = []
    for pair in json.loads(a.pairs.read_text()):
        x = to_signed(read_image(a.data_root / pair["source"], "cuda", multiple=8))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        y = deblurrer(x, a.refiner_tile, a.refiner_overlap)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        save_image(to_unit_range(y), a.out / f"{pair['index']:04d}.png")
    warm = times[3:] or times
    print(f"{a.pairs.stem}: {len(times)} images, mean time {sum(warm) / len(warm):.3f}s"
          + (" (excluding the first 3)" if len(times) > 3 else ""))


if __name__ == "__main__":
    main()
