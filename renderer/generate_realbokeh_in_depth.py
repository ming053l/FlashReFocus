import argparse
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEPTH_ANYTHING_ROOT = Path(__file__).resolve().parents[2] / "Depth-Anything-3" / "src"
DEPTH_ANYTHING_FALLBACK_ROOT = (
    Path(__file__).resolve().parents[2] / "old" / "IGDDD"
)
DEPTH_PRO_ROOT = Path(__file__).resolve().parents[2] / "ml-depth-pro"


def load_depth_model(model_name, device):
    depthanything_root = DEPTH_ANYTHING_ROOT
    if not (depthanything_root / "depth_anything_3" / "api.py").is_file():
        depthanything_root = DEPTH_ANYTHING_FALLBACK_ROOT
    if not (depthanything_root / "depth_anything_3" / "api.py").is_file():
        raise FileNotFoundError(
            "DepthAnythingV3 package not found under "
            f"{DEPTH_ANYTHING_ROOT} or {DEPTH_ANYTHING_FALLBACK_ROOT}"
        )

    sys.modules.pop("api", None)
    if str(depthanything_root) not in sys.path:
        sys.path.insert(0, str(depthanything_root))
    if "moviepy.editor" not in sys.modules:
        try:
            import moviepy

            sys.modules["moviepy.editor"] = moviepy
        except ModuleNotFoundError:
            pass
    if "depth_anything_3.utils.export" not in sys.modules:
        export_module = types.ModuleType("depth_anything_3.utils.export")

        def _disabled_export(*args, **kwargs):
            raise RuntimeError("DepthAnything export is disabled in this generator.")

        export_module.export = _disabled_export
        sys.modules["depth_anything_3.utils.export"] = export_module
    if "depth_anything_3.utils.pose_align" not in sys.modules:
        pose_align_module = types.ModuleType("depth_anything_3.utils.pose_align")

        def _disabled_pose_align(*args, **kwargs):
            raise RuntimeError(
                "DepthAnything pose alignment is disabled in this generator."
            )

        pose_align_module.align_poses_umeyama = _disabled_pose_align
        pose_align_module.batch_align_poses_umeyama = _disabled_pose_align
        sys.modules["depth_anything_3.utils.pose_align"] = pose_align_module

    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    model.eval()
    print(f"Loaded DepthAnythingV3 model: {model_name}")
    print(f"Device: {device}")
    return model


def load_depth_pro_model(depth_pro_root, checkpoint, fp32, device):
    src_root = depth_pro_root / "src"
    checkpoint = (
        checkpoint
        if checkpoint is not None
        else depth_pro_root / "checkpoints" / "depth_pro.pt"
    )
    if not (src_root / "depth_pro").is_dir():
        raise FileNotFoundError(f"Depth Pro package not found under {src_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Depth Pro checkpoint not found: {checkpoint}. "
            f"Run `{depth_pro_root / 'get_pretrained_models.sh'}` first."
        )

    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    if "pillow_heif" not in sys.modules:
        try:
            import pillow_heif  # noqa: F401
        except ModuleNotFoundError:
            pillow_heif_stub = types.ModuleType("pillow_heif")

            def _missing_heif(*args, **kwargs):
                raise ModuleNotFoundError(
                    "pillow_heif is required for HEIF/HEIC Depth Pro inputs."
                )

            pillow_heif_stub.register_heif_opener = lambda *args, **kwargs: None
            pillow_heif_stub.open_heif = _missing_heif
            sys.modules["pillow_heif"] = pillow_heif_stub

    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    precision = torch.float32 if fp32 or device.type != "cuda" else torch.float16
    config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint))
    model, transform = depth_pro.create_model_and_transforms(
        config=config,
        device=device,
        precision=precision,
    )
    model.eval()
    print(f"Loaded Depth Pro checkpoint: {checkpoint}")
    print(f"Device: {device}")
    return model, transform, depth_pro


def iter_source_images(realbokeh_root, splits):
    for split in splits:
        split_dir = realbokeh_root / split
        metadata_dir = split_dir / "metadata"
        if not metadata_dir.exists():
            print(f"[WARN] metadata directory not found: {metadata_dir}")
            continue

        seen = set()
        for metadata_path in sorted(metadata_dir.glob("*.json")):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # RealBokeh depth maps should be generated from source images under in/.
            rel_source = Path(metadata["source_image"])
            if rel_source.parts[0] != "in":
                print(f"[WARN] source_image is not under in/: {metadata_path}")
            source_path = split_dir / rel_source
            if source_path in seen:
                continue
            seen.add(source_path)

            output_path = split_dir / "in_depth" / f"{source_path.stem}.tiff"
            yield split, source_path, output_path


def save_depth_image(depth, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth = depth.astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    Image.fromarray(depth, mode="F").save(output_path)


def infer_depth_batch(model, image_paths, process_res):
    prediction = model.inference(
        [str(path) for path in image_paths],
        process_res=process_res,
    )
    depths = prediction.depth
    if len(depths) != len(image_paths):
        raise RuntimeError(
            f"DepthAnything returned {len(depths)} maps for {len(image_paths)} images"
        )
    return depths


def infer_depth_pro_image(model, transform, depth_pro_module, image_path, auto_rotate):
    image, _, f_px = depth_pro_module.load_rgb(image_path, auto_rotate=auto_rotate)
    prediction = model.infer(transform(image), f_px=f_px)
    return prediction["depth"].detach().cpu().numpy().squeeze().astype(np.float32)


def convert_depth_domain(depth, output_domain):
    depth = depth.astype(np.float32)
    if output_domain == "depth":
        return depth
    if output_domain == "inverse-depth":
        return 1.0 / np.clip(depth, 1e-6, None)
    raise ValueError(f"Unsupported output domain: {output_domain}")


def process_realbokeh(
    realbokeh_root,
    splits,
    model,
    estimator,
    process_res,
    batch_size,
    overwrite,
    output_domain,
    depth_pro_transform=None,
    depth_pro_module=None,
    depth_pro_auto_rotate=False,
):
    samples = list(iter_source_images(realbokeh_root, splits))
    if not samples:
        raise FileNotFoundError(f"No RealBokeh source images found under {realbokeh_root}")

    processed = 0
    skipped = 0
    batch = []

    def flush_batch():
        nonlocal processed
        if not batch:
            return

        batch_paths = [sample[1] for sample in batch]
        if estimator == "da3":
            depths = infer_depth_batch(model, batch_paths, process_res=process_res)
        elif estimator == "depth-pro":
            depths = [
                infer_depth_pro_image(
                    model,
                    depth_pro_transform,
                    depth_pro_module,
                    image_path,
                    auto_rotate=depth_pro_auto_rotate,
                )
                for image_path in batch_paths
            ]
        else:
            raise ValueError(f"Unsupported depth estimator: {estimator}")
        for (_, source_path, output_path), depth in zip(batch, depths):
            with Image.open(source_path) as image:
                original_size = image.size

            depth = convert_depth_domain(depth, output_domain)
            if depth.shape != (original_size[1], original_size[0]):
                depth_image = Image.fromarray(depth, mode="F")
                depth_image = depth_image.resize(
                    original_size,
                    resample=Image.Resampling.BILINEAR,
                )
                depth = np.asarray(depth_image, dtype=np.float32)

            save_depth_image(depth, output_path)
            processed += 1

            if processed % 50 == 0:
                print(f"processed={processed}, skipped={skipped}")

        batch.clear()

    for split, source_path, output_path in samples:
        if not source_path.exists():
            print(f"[WARN] source image not found: {source_path}")
            skipped += 1
            continue
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        batch.append((split, source_path, output_path))
        if len(batch) >= batch_size:
            flush_batch()

    flush_batch()
    print(
        f"Done: root={realbokeh_root}, splits={','.join(splits)}, "
        f"processed={processed}, skipped={skipped}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate RealBokeh in_depth maps with DA3 or Depth Pro."
    )
    parser.add_argument(
        "--realbokeh-root",
        type=Path,
        default=Path("/ssddd/chingheng/Deblurring-Dataset/RealBokeh"),
        help="RealBokeh root containing train/validation/test splits.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--depth-estimator",
        type=str,
        default="da3",
        choices=["da3", "depth-pro"],
        help="Depth estimator used to regenerate in_depth maps.",
    )
    parser.add_argument(
        "--depth-model-name",
        type=str,
        default="depth-anything/da3metric-large",
        help="DepthAnythingV3 model name or local checkpoint.",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=1024,
        help="DepthAnythingV3 inference resolution.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of images per depth generation batch. Depth Pro is still "
            "inferred one image at a time inside each batch."
        ),
    )
    parser.add_argument(
        "--depth-pro-root",
        type=Path,
        default=DEPTH_PRO_ROOT,
        help="Path to the ml-depth-pro repository.",
    )
    parser.add_argument(
        "--depth-pro-checkpoint",
        type=Path,
        default=None,
        help=(
            "Depth Pro checkpoint. Defaults to "
            "DEPTH_PRO_ROOT/checkpoints/depth_pro.pt."
        ),
    )
    parser.add_argument(
        "--depth-pro-fp32",
        action="store_true",
        help="Run Depth Pro in float32 instead of float16 on CUDA.",
    )
    parser.add_argument(
        "--depth-pro-auto-rotate",
        action="store_true",
        help=(
            "Let Depth Pro apply EXIF orientation before inference. Disabled by "
            "default so generated maps align with Bokehlicious loaders, which "
            "read RealBokeh images without EXIF transpose."
        ),
    )
    parser.add_argument(
        "--output-domain",
        type=str,
        default="depth",
        choices=["depth", "inverse-depth"],
        help=(
            "Scalar field saved as float32 TIFF. Depth Pro returns metric depth; "
            "inverse-depth writes inverse metric depth."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing in_depth files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if not args.realbokeh_root.exists():
        raise FileNotFoundError(f"RealBokeh root not found: {args.realbokeh_root}")

    device = torch.device(args.device)
    depth_pro_transform = None
    depth_pro_module = None
    if args.depth_estimator == "da3":
        model = load_depth_model(args.depth_model_name, device=device)
    elif args.depth_estimator == "depth-pro":
        model, depth_pro_transform, depth_pro_module = load_depth_pro_model(
            depth_pro_root=args.depth_pro_root,
            checkpoint=args.depth_pro_checkpoint,
            fp32=args.depth_pro_fp32,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported depth estimator: {args.depth_estimator}")

    with torch.no_grad():
        process_realbokeh(
            realbokeh_root=args.realbokeh_root,
            splits=args.splits,
            model=model,
            estimator=args.depth_estimator,
            process_res=args.process_res,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            output_domain=args.output_domain,
            depth_pro_transform=depth_pro_transform,
            depth_pro_module=depth_pro_module,
            depth_pro_auto_rotate=args.depth_pro_auto_rotate,
        )


if __name__ == "__main__":
    main()
