"""Cache Depth Anything V2 output for the RealBokeh source images as float32 TIFF (<split>/da2_depth/<stem>.tiff), read
by render_data.RealBokehPairs for renderer adaptation. The cache was made at input size 518 in fp32, as below; at inference the
renderer runs Depth Anything V2 itself (input size 1024, fp16).

  python renderer/generate_da2_depth.py [--root <data>/RealBokeh] [--splits train validation]"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
from infer_single_focus import load_da2_model  # noqa: E402


def sources(root, split):
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
    ap.add_argument("--checkpoint", type=Path, default=paths.DA2)
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    model = load_da2_model("vitl", a.checkpoint, torch.device("cuda"))
    done = 0
    with torch.inference_mode():
        for split in a.splits:
            for source, out in sources(a.root, split):
                if out.exists() and not a.overwrite:
                    continue
                depth = model.infer_image(cv2.imread(str(source), cv2.IMREAD_COLOR), input_size=a.input_size)
                out.parent.mkdir(parents=True, exist_ok=True)
                depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                Image.fromarray(depth, mode="F").save(out)
                done += 1
    print(f"wrote {done} depth maps")


if __name__ == "__main__":
    main()
