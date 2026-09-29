import argparse
import json
from pathlib import Path


BOKEHLICIOUS_ROOT = Path(__file__).resolve().parent
DA2_ROOT = BOKEHLICIOUS_ROOT.parent / "Depth-Anything-V2"
DEFAULT_REALBOKEH_ROOT = Path("/ssddd/chingheng/Deblurring-Dataset/RealBokeh")
DEFAULT_EBB_ROOT = Path("/ssddd/chingheng/Deblurring-Dataset/EBB")
DEFAULT_REAL3C_ROOT = BOKEHLICIOUS_ROOT / "Real3C"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


def image_sort_key(path: Path):
    return int(path.stem) if path.stem.isdigit() else path.stem


def save_float_tiff(depth, output_path: Path):
    import numpy as np
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth = depth.astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    Image.fromarray(depth, mode="F").save(output_path)


def load_da2_model(da2_root: Path, checkpoint: Path, encoder: str, device):
    import torch

    if not (da2_root / "depth_anything_v2" / "dpt.py").is_file():
        raise FileNotFoundError(f"Depth-Anything-V2 repo not found: {da2_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Download the V2 checkpoint, for example depth_anything_v2_vitl.pth, "
            "and pass it with --checkpoint or put it under "
            f"{da2_root / 'checkpoints'}."
        )

    import sys

    if str(da2_root) not in sys.path:
        sys.path.insert(0, str(da2_root))

    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    if encoder not in model_configs:
        raise ValueError(f"Unsupported encoder: {encoder}")

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device).eval()
    print(f"Loaded Depth Anything V2: encoder={encoder}, checkpoint={checkpoint}")
    print(f"Device: {device}")
    return model


def infer_one(model, image_path: Path, input_size: int):
    import cv2
    import numpy as np

    raw_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw_image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return model.infer_image(raw_image, input_size=input_size).astype(np.float32)


def iter_folder_images(image_dir: Path):
    if not image_dir.is_dir():
        print(f"[WARN] image directory not found: {image_dir}")
        return
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES]
    for path in sorted(paths, key=image_sort_key):
        yield path


def iter_realbokeh_sources(realbokeh_root: Path, splits: list[str]):
    for split in splits:
        split_dir = realbokeh_root / split
        metadata_dir = split_dir / "metadata"
        if not metadata_dir.is_dir():
            print(f"[WARN] RealBokeh metadata directory not found: {metadata_dir}")
            continue

        seen = set()
        for metadata_path in sorted(metadata_dir.glob("*.json")):
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            source_path = split_dir / metadata["source_image"]
            if source_path in seen:
                continue
            seen.add(source_path)
            output_path = split_dir / "da2_depth" / f"{source_path.stem}.tiff"
            yield source_path, output_path


def iter_ebb_sources(ebb_root: Path, split: str):
    image_dir = ebb_root / split / "original"
    output_dir = ebb_root / split / "da2_depth"
    for source_path in iter_folder_images(image_dir):
        yield source_path, output_dir / f"{source_path.stem}.tiff"


def iter_real3c_sources(real3c_root: Path, split: str):
    split_dir = real3c_root / split
    candidates = [split_dir / "original"]
    for image_dir in candidates:
        paths = list(iter_folder_images(image_dir))
        if paths:
            for source_path in paths:
                yield source_path, split_dir / "da2_depth" / f"{source_path.stem}.tiff"
            return
    print(f"[WARN] No Real3C source images found under: {split_dir / 'original'}")


def process_samples(model, samples, input_size: int, overwrite: bool, label: str):
    samples = list(samples)
    if not samples:
        print(f"[WARN] No samples for {label}")
        return 0, 0, 0

    processed = 0
    skipped = 0
    missing = 0
    total = len(samples)
    print(f"{label}: {total} candidate images")

    for idx, (source_path, output_path) in enumerate(samples, start=1):
        if not source_path.is_file():
            print(f"[WARN] source image not found: {source_path}")
            missing += 1
            continue
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        depth = infer_one(model, source_path, input_size=input_size)
        save_float_tiff(depth, output_path)
        processed += 1

        if processed == 1 or processed % 50 == 0:
            print(
                f"{label}: processed={processed}, skipped={skipped}, "
                f"missing={missing}, current={idx}/{total}"
            )

    print(f"{label}: done processed={processed}, skipped={skipped}, missing={missing}")
    return processed, skipped, missing


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate float32 TIFF da2_depth maps with Depth Anything V2 relative "
            "disparity-like output."
        )
    )
    parser.add_argument("--da2-root", type=Path, default=DA2_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Depth Anything V2 checkpoint. Defaults to <da2-root>/checkpoints/depth_anything_v2_<encoder>.pth.",
    )
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vitl")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--device",
        default="auto",
        help='Device for inference. Use "auto", "cuda", or "cpu".',
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--realbokeh-root", type=Path, default=DEFAULT_REALBOKEH_ROOT)
    parser.add_argument("--realbokeh-splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--ebb-root", type=Path, default=DEFAULT_EBB_ROOT)
    parser.add_argument("--ebb-split", default="validation")
    parser.add_argument("--real3c-root", type=Path, default=DEFAULT_REAL3C_ROOT)
    parser.add_argument("--real3c-split", default="validation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["realbokeh", "ebb", "real3c"],
        default=["realbokeh", "ebb", "real3c"],
        help="Datasets to process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = args.da2_root / "checkpoints" / f"depth_anything_v2_{args.encoder}.pth"

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = load_da2_model(args.da2_root, checkpoint, args.encoder, device)

    totals = {"processed": 0, "skipped": 0, "missing": 0}
    with torch.inference_mode():
        if "realbokeh" in args.datasets:
            result = process_samples(
                model,
                iter_realbokeh_sources(args.realbokeh_root, args.realbokeh_splits),
                input_size=args.input_size,
                overwrite=args.overwrite,
                label="RealBokeh",
            )
            totals["processed"] += result[0]
            totals["skipped"] += result[1]
            totals["missing"] += result[2]

        if "ebb" in args.datasets:
            result = process_samples(
                model,
                iter_ebb_sources(args.ebb_root, args.ebb_split),
                input_size=args.input_size,
                overwrite=args.overwrite,
                label="EBB",
            )
            totals["processed"] += result[0]
            totals["skipped"] += result[1]
            totals["missing"] += result[2]

        if "real3c" in args.datasets:
            result = process_samples(
                model,
                iter_real3c_sources(args.real3c_root, args.real3c_split),
                input_size=args.input_size,
                overwrite=args.overwrite,
                label="Real3C",
            )
            totals["processed"] += result[0]
            totals["skipped"] += result[1]
            totals["missing"] += result[2]

    print(
        "All done: "
        f"processed={totals['processed']}, skipped={totals['skipped']}, "
        f"missing={totals['missing']}"
    )


if __name__ == "__main__":
    main()
