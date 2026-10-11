#!/usr/bin/env python
"""Cache Depth Anything V2 output for the RealBokeh source images as float32 TIFF (<split>/da2_depth/<stem>.tiff), read
by training/rendering_data.RealBokehPairs for renderer adaptation. The cache is made at input size 518 in fp32, as
for the released renderer; at inference the depth estimator runs at input size 1024 in fp16.

  python scripts/cache_depth.py [--root <data>/RealBokeh] [--splits train validation]"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from flashrefocus import paths
from flashrefocus.rendering.depth import load_depth_anything_v2


def source_images(root, split):
    seen = set()
    for meta_path in sorted((root / split / "metadata").glob("*.json")):
        source = root / split / json.loads(meta_path.read_text())["source_image"]
        if source not in seen:
            seen.add(source)
            yield source, root / split / "da2_depth" / f"{source.stem}.tiff"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=paths.DATA / "RealBokeh")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--checkpoint", type=Path, default=paths.DEPTH_ANYTHING_V2)
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    model = load_depth_anything_v2("vitl", a.checkpoint, "cuda")
    written = 0
    with torch.inference_mode():
        for split in a.splits:
            for source, out in source_images(a.root, split):
                if out.exists() and not a.overwrite:
                    continue
                disparity = model.infer_image(cv2.imread(str(source), cv2.IMREAD_COLOR), input_size=a.input_size)
                out.parent.mkdir(parents=True, exist_ok=True)
                disparity = np.nan_to_num(disparity.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                Image.fromarray(disparity, mode="F").save(out)
                written += 1
    print(f"wrote {written} depth maps")


if __name__ == "__main__":
    main()
