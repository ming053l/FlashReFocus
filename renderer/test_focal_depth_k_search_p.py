import argparse
import csv
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from dataset.loader_ebb import EBBFocalDepth, ebb_image_paths
from dataset.loader_focal_depth import RealBokehFocalDepth
from dataset.util import Mode
from method.config import bokehlicious_size_builder
from method.model_dma_p import Bokehlicious
from train import (
    MetricTracker,
    build_metric_funcs as build_train_metric_funcs,
    extract_model_state_dict,
)

DEPTH_ANYTHING_ROOT = Path("/ssddd/chingheng/ImageStitching/Defocus/old/IGDDD")
DEPTH_PRO_ROOT = Path("/ssddd/chingheng/ImageStitching/Defocus/ml-depth-pro")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Bokehlicious DMA-p on focal-depth validation data by "
            "searching K embeddings and selecting the per-sample best SSIM."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ebb",
        choices=["realbokeh", "ebb"],
        help="Validation dataset type.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/ssddd/chingheng/Deblurring-Dataset/EBB"),
        help=(
            "Dataset root. For realbokeh: root containing validation/in, "
            "validation/in_depth, and metadata. For ebb: root containing "
            "validation/original and validation/bokeh."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=Path("checkpoints/train_v6_realbokeh/best.pt"),
        help="Path to the trained DMA checkpoint.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "large", "defocus_deblur"],
        help="Bokehlicious model size/config.",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help="Override validation focus annotation JSON. Defaults to validation/focus_points_f2.json.",
    )
    parser.add_argument(
        "--depth-map-dir",
        type=str,
        default="in_depth",
        help=(
            "Depth map directory inside the validation split. For --dataset ebb, "
            "this is used as the DA3 cache directory."
        ),
    )
    parser.add_argument(
        "--target-apertures",
        type=str,
        default="all",
        choices=["all", "annotated"],
        help="Evaluate all metadata targets or only the annotated f2 target.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--val-metrics",
        type=str,
        default="psnr,ssim,lpips,dists",
        help=(
            "Comma-separated metrics to report at the best-SSIM K: "
            "psnr,ssim,lpips,dists."
        ),
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone used by torchmetrics LPIPS.",
    )
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=100)
    parser.add_argument(
        "--k-normalizer",
        type=float,
        default=100.0,
        help="K embedding is clamp(K / k_normalizer, 0, 1).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Load checkpoint with strict=True. Default is strict=False.",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Per-sample CSV output path. Defaults to OUT_DIR/results.csv.",
    )
    parser.add_argument(
        "--save-summary",
        type=Path,
        default=None,
        help="CSV output path for averaged metrics. Defaults to OUT_DIR/summary.csv.",
    )
    parser.add_argument(
        "--no-save-outputs",
        action="store_true",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/focal_depth_k_search"),
        help="Output directory for CSV files and the images subdirectory.",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "jpg"],
    )
    parser.add_argument("--use-mres-after-in-stage2", action="store_true")
    parser.add_argument("--use-mres-before-out-stage", action="store_true")
    parser.add_argument("--mres-num-blocks", type=int, default=2)
    parser.add_argument("--attn-version", type=str, default="v1")
    parser.add_argument(
        "--disable-radiance-visualization",
        action="store_true",
        help="Instantiate the model with the radiance visualization branch disabled.",
    )
    parser.add_argument(
        "--depth-estimator",
        type=str,
        default="da3",
        choices=["da3", "depth-pro"],
        help=(
            "Depth estimator used to generate missing cached depth maps. "
            "realbokeh can also use precomputed maps via --depth-map-dir."
        ),
    )
    parser.add_argument(
        "--depth-model-name",
        type=str,
        default="depth-anything/da3metric-large",
        help="DepthAnythingV3 model name or local checkpoint for --dataset ebb.",
    )
    parser.add_argument(
        "--depth-process-res",
        type=int,
        default=1024,
        help="DepthAnythingV3 inference resolution.",
    )
    parser.add_argument(
        "--depth-batch-size",
        type=int,
        default=1,
        help="Number of images per depth generation batch.",
    )
    parser.add_argument(
        "--depth-normalize-scale",
        type=float,
        default=20.0,
        help=("Fixed metric depth scale used by focal-depth datasets."),
    )
    parser.add_argument(
        "--depth-mode",
        type=str,
        default="auto",
        choices=[
            "auto",
            "metric_scaled",
            "metric_minmax",
            "metric_inverse",
            "disparity",
            "raw",
        ],
        help=(
            "Depth preprocessing mode. auto maps da2_depth/midas_disparity to "
            "disparity and metric depth dirs to metric_scaled."
        ),
    )
    parser.add_argument(
        "--inverse-depth",
        action="store_true",
        help=(
            "Convert cached depth maps to per-image normalized inverse depth before "
            "building focal-depth conditioning."
        ),
    )
    parser.add_argument(
        "--focus-map-type",
        type=str,
        default="cosine",
        choices=["cosine", "gaussian", "gaussain"],
        help="Point focus map shape used by focal-depth datasets.",
    )
    parser.add_argument(
        "--focus-map-gaussian-sigma",
        type=float,
        default=None,
        help="Absolute gaussian sigma for --focus-map-type gaussian.",
    )
    parser.add_argument(
        "--focus-map-gaussian-sigma-scale",
        type=float,
        default=6.0,
        help="Image-size divisor used to derive gaussian sigma when sigma is unset.",
    )
    parser.add_argument(
        "--overwrite-depth",
        action="store_true",
        help="Regenerate cached depth maps.",
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
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(args):
    config = bokehlicious_size_builder(args.size)
    return Bokehlicious(
        **config,
        use_mres_after_in_stage2=args.use_mres_after_in_stage2,
        use_mres_before_out_stage=args.use_mres_before_out_stage,
        mres_num_blocks=args.mres_num_blocks,
        attn_version=args.attn_version,
        use_radiance_visualization=not args.disable_radiance_visualization,
    )


def load_weights(model, checkpoint_path, device, strict=False):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_model_state_dict(checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }

    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(f"Loaded weights from {checkpoint_path} (strict={strict})")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True).float()
        else:
            moved[key] = value
    return moved


def k_to_embedding(k_value, normalizer, device, dtype, batch_shape):
    embedding = max(0.0, min(float(k_value) / float(normalizer), 1.0))
    return torch.full(batch_shape, embedding, device=device, dtype=dtype)


def metric_value(metric_name, metric_fn, outputs, targets):
    if isinstance(metric_fn, nn.Module):
        value = metric_fn(outputs.clamp(0.0, 1.0), targets.clamp(0.0, 1.0)).item()
        metric_fn.reset()
        return value
    return metric_fn(outputs, targets).item()


def normalize_names(image_names):
    if isinstance(image_names, str):
        return [image_names]
    if isinstance(image_names, tuple):
        return list(image_names)
    return image_names


def save_outputs(outputs, image_names, best_ks, out_dir, image_format):
    out_dir.mkdir(parents=True, exist_ok=True)
    for output, image_name, best_k in zip(
        outputs, normalize_names(image_names), best_ks
    ):
        stem = Path(str(image_name)).stem
        out_path = out_dir / f"{stem}_bestK{int(best_k):02d}.{image_format}"
        to_pil_image(output.detach().cpu().clamp(0.0, 1.0)).save(out_path)


def build_metric_funcs(metric_names, lpips_net):
    funcs = {}
    for name in metric_names:
        if name == "dists":
            funcs[name] = DeepImageStructureAndTextureSimilarity()
        else:
            funcs[name] = build_train_metric_funcs([name], lpips_net)[name]
    return funcs


def evaluate_k_value(
    model,
    batch,
    inputs,
    targets,
    metric_funcs,
    k_value,
    k_normalizer,
    device,
    use_amp,
):
    bokeh_strength = k_to_embedding(
        k_value=k_value,
        normalizer=k_normalizer,
        device=device,
        dtype=inputs.dtype,
        batch_shape=batch["bokeh_strength"].shape,
    )
    with torch.amp.autocast(
        device_type=device.type,
        enabled=use_amp,
        dtype=torch.bfloat16,
    ):
        outputs = model(
            source=inputs,
            bokeh_strength=bokeh_strength,
            focus_map=batch["focus_map"],
            depth_map=batch.get("depth_map"),
            disp_focus=batch.get("disp_focus"),
        )

    sample_ssim = metric_funcs["ssim"](outputs, targets)
    if sample_ssim.dim() > 0:
        sample_ssim = sample_ssim[0]
    return sample_ssim.item(), outputs


def binary_search_best_k(
    model,
    batch,
    inputs,
    targets,
    metric_funcs,
    k_values,
    k_normalizer,
    device,
    use_amp,
):
    cached_results = {}

    def get_result(index):
        if index not in cached_results:
            cached_results[index] = evaluate_k_value(
                model=model,
                batch=batch,
                inputs=inputs,
                targets=targets,
                metric_funcs=metric_funcs,
                k_value=k_values[index],
                k_normalizer=k_normalizer,
                device=device,
                use_amp=use_amp,
            )
        return cached_results[index]

    lo = 0
    hi = len(k_values) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        mid_ssim, _ = get_result(mid)
        next_ssim, _ = get_result(mid + 1)
        if mid_ssim < next_ssim:
            lo = mid + 1
        else:
            hi = mid

    best_ssim, best_outputs = get_result(lo)
    best_k = float(k_values[lo])
    return best_k, best_ssim, best_outputs.detach().clone()


def load_depth_model(model_name, device):
    depthanything_root = DEPTH_ANYTHING_ROOT
    if depthanything_root.name == "depth_anything_3":
        depthanything_root = depthanything_root.parent
    if not (depthanything_root / "depth_anything_3" / "api.py").is_file():
        raise FileNotFoundError(
            f"DepthAnythingV3 package not found under {depthanything_root}"
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
            raise RuntimeError("DepthAnything export is disabled in this evaluator.")

        export_module.export = _disabled_export
        sys.modules["depth_anything_3.utils.export"] = export_module
    if "depth_anything_3.utils.pose_align" not in sys.modules:
        pose_align_module = types.ModuleType("depth_anything_3.utils.pose_align")

        def _disabled_pose_align(*args, **kwargs):
            raise RuntimeError(
                "DepthAnything pose alignment is disabled in this evaluator."
            )

        pose_align_module.align_poses_umeyama = _disabled_pose_align
        pose_align_module.batch_align_poses_umeyama = _disabled_pose_align
        sys.modules["depth_anything_3.utils.pose_align"] = pose_align_module
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    model.eval()
    print(f"Loaded DepthAnythingV3 model: {model_name}")
    return model


def save_metric_depth_image(depth, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth = depth.astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    Image.fromarray(depth, mode="F").save(output_path)


def infer_depth_batch(model, image_paths, process_res):
    prediction = model.inference(
        [str(path) for path in image_paths], process_res=process_res
    )
    depths = prediction.depth
    if len(depths) != len(image_paths):
        raise RuntimeError(
            f"DepthAnything returned {len(depths)} maps for {len(image_paths)} images"
        )
    return depths


def load_depth_pro_model(args, device):
    depth_pro_root = args.depth_pro_root
    src_root = depth_pro_root / "src"
    checkpoint = (
        args.depth_pro_checkpoint
        if args.depth_pro_checkpoint is not None
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

    precision = (
        torch.float32 if args.depth_pro_fp32 or device.type != "cuda" else torch.float16
    )
    config = replace(
        DEFAULT_MONODEPTH_CONFIG_DICT,
        checkpoint_uri=str(checkpoint),
    )
    model, transform = depth_pro.create_model_and_transforms(
        config=config,
        device=device,
        precision=precision,
    )
    model.eval()
    print(f"Loaded Depth Pro checkpoint: {checkpoint}")
    return model, transform, depth_pro


def infer_depth_pro_image(model, transform, depth_pro_module, image_path):
    image, _, f_px = depth_pro_module.load_rgb(image_path)
    prediction = model.infer(transform(image), f_px=f_px)
    return prediction["depth"].detach().cpu().numpy().squeeze().astype(np.float32)


def realbokeh_source_image_paths(data_dir, mode):
    split_dir = data_dir / mode.value
    metadata_dir = split_dir / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(
            f"RealBokeh metadata directory not found: {metadata_dir}"
        )

    source_paths = []
    seen = set()
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        import json

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        source_path = split_dir / metadata["source_image"]
        if source_path not in seen:
            source_paths.append(source_path)
            seen.add(source_path)
    return source_paths


def ensure_depth_maps(
    args,
    device,
    image_paths,
    depth_dir,
    desc_prefix,
):
    missing = [
        path
        for path in image_paths
        if args.overwrite_depth or not (depth_dir / f"{path.stem}.tiff").exists()
    ]
    if not missing:
        print(f"Using cached {args.depth_estimator} depth maps from {depth_dir}")
        return

    if args.depth_batch_size < 1:
        raise ValueError("--depth-batch-size must be >= 1")
    print(
        f"Generating {len(missing)} {desc_prefix} {args.depth_estimator} "
        f"depth maps into {depth_dir} "
        f"({len(image_paths) - len(missing)} cached)"
    )
    if args.depth_estimator == "da3":
        depth_model = load_depth_model(args.depth_model_name, device=device)
        depth_pro_transform = None
        depth_pro_module = None
    elif args.depth_estimator == "depth-pro":
        depth_model, depth_pro_transform, depth_pro_module = load_depth_pro_model(
            args, device=device
        )
    else:
        raise ValueError(f"Unsupported depth estimator: {args.depth_estimator}")

    batch = []

    def flush_batch(model):
        if not batch:
            return
        if args.depth_estimator == "da3":
            depths = infer_depth_batch(model, batch, process_res=args.depth_process_res)
        else:
            depths = [
                infer_depth_pro_image(
                    model,
                    depth_pro_transform,
                    depth_pro_module,
                    image_path,
                )
                for image_path in batch
            ]
        for image_path, depth in zip(batch, depths):
            with Image.open(image_path) as image:
                original_size = image.size

            depth = depth.astype(np.float32)
            if depth.shape != (original_size[1], original_size[0]):
                depth_image = Image.fromarray(depth, mode="F")
                depth_image = depth_image.resize(
                    original_size,
                    resample=Image.Resampling.BILINEAR,
                )
                depth = np.asarray(depth_image, dtype=np.float32)

            save_metric_depth_image(depth, depth_dir / f"{image_path.stem}.tiff")
        batch.clear()

    with torch.no_grad():
        for image_path in tqdm(missing, desc=f"{desc_prefix}-{args.depth_estimator}"):
            batch.append(image_path)
            if len(batch) >= args.depth_batch_size:
                flush_batch(depth_model)
        flush_batch(depth_model)
    del depth_model
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate_k_search(
    model,
    loader,
    device,
    metric_funcs,
    k_values,
    k_normalizer,
    amp=False,
    save_outputs_flag=False,
    out_dir=None,
    image_format="png",
):
    model.eval()
    for metric_fn in metric_funcs.values():
        if isinstance(metric_fn, nn.Module):
            metric_fn.to(device)
            metric_fn.eval()

    tracker = MetricTracker(*metric_funcs.keys())
    rows = []
    use_amp = bool(amp and device.type == "cuda")

    for batch in tqdm(loader, desc="val-k-search"):
        batch = move_batch_to_device(batch, device)
        inputs = batch["source"]
        targets = batch["target"]
        batch_size = inputs.size(0)

        best_k, best_ssim, best_outputs = binary_search_best_k(
            model=model,
            batch=batch,
            inputs=inputs,
            targets=targets,
            metric_funcs=metric_funcs,
            k_values=k_values,
            k_normalizer=k_normalizer,
            device=device,
            use_amp=use_amp,
        )
        best_ks = torch.full((batch_size,), best_k, device=device)

        metric_results = {}
        for metric_name, metric_fn in metric_funcs.items():
            metric_results[metric_name] = metric_value(
                metric_name, metric_fn, best_outputs, targets
            )
            tracker.update(metric_name, metric_results[metric_name], n=batch_size)

        image_names = normalize_names(batch["image_name"])
        scene_ids = batch.get("scene_id", [""] * batch_size)
        if isinstance(scene_ids, str):
            scene_ids = [scene_ids]

        for i, image_name in enumerate(image_names):
            row = {
                "image_name": str(image_name),
                "scene_id": str(scene_ids[i]) if i < len(scene_ids) else "",
                "best_k": int(best_ks[i].item()),
                "best_k_embedding": best_ks[i].item() / float(k_normalizer),
            }
            row.update({name: value for name, value in metric_results.items()})
            rows.append(row)

        if save_outputs_flag:
            save_outputs(best_outputs, image_names, best_ks, out_dir, image_format)

    return tracker.result(), rows


def write_csv(path, rows, metric_names, results=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "scene_id",
        "best_k",
        "best_k_embedding",
        *metric_names,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        if results is not None:
            mean_row = {
                "image_name": "MEAN",
                "scene_id": "",
                "best_k": (
                    sum(float(row["best_k"]) for row in rows) / len(rows)
                    if rows
                    else ""
                ),
                "best_k_embedding": (
                    sum(float(row["best_k_embedding"]) for row in rows) / len(rows)
                    if rows
                    else ""
                ),
            }
            mean_row.update({name: results.get(name, "") for name in metric_names})
            writer.writerow(mean_row)


def write_summary_csv(path, rows, metric_names, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_best_k = sum(float(row["best_k"]) for row in rows) / len(rows) if rows else ""
    mean_best_k_embedding = (
        sum(float(row["best_k_embedding"]) for row in rows) / len(rows) if rows else ""
    )
    summary = {
        "num_samples": len(rows),
        "mean_best_k": mean_best_k,
        "mean_best_k_embedding": mean_best_k_embedding,
    }
    summary.update({f"mean_{name}": results.get(name, "") for name in metric_names})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main():
    args = parse_args()
    if "ssim" not in {m.strip().lower() for m in args.val_metrics.split(",")}:
        args.val_metrics = args.val_metrics.rstrip(",") + ",ssim"
        print("Added ssim to --val-metrics because K search selects by SSIM.")
    if args.k_min > args.k_max:
        raise ValueError("--k-min must be <= --k-max")
    if args.k_normalizer <= 0:
        raise ValueError("--k-normalizer must be > 0")
    if args.depth_normalize_scale <= 0:
        raise ValueError("--depth-normalize-scale must be > 0")
    if args.focus_map_gaussian_sigma is not None and args.focus_map_gaussian_sigma <= 0:
        raise ValueError("--focus-map-gaussian-sigma must be > 0")
    if args.focus_map_gaussian_sigma_scale <= 0:
        raise ValueError("--focus-map-gaussian-sigma-scale must be > 0")
    if args.batch_size != 1:
        raise ValueError(
            "This script selects best K by per-sample SSIM, so --batch-size must be 1."
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
    if args.save_csv is None:
        args.save_csv = args.out_dir / "results.csv"
    if args.save_summary is None:
        args.save_summary = args.out_dir / "summary.csv"
    image_out_dir = args.out_dir / "images"

    device = resolve_device(args.device)
    if args.dataset == "realbokeh":
        if args.depth_estimator == "depth-pro" and args.depth_map_dir == "in_depth":
            args.depth_map_dir = "depth_pro_depth"
        if args.depth_estimator == "depth-pro" or args.overwrite_depth:
            depth_dir = args.data_dir / Mode.VAL.value / args.depth_map_dir
            ensure_depth_maps(
                args=args,
                device=device,
                image_paths=realbokeh_source_image_paths(args.data_dir, Mode.VAL),
                depth_dir=depth_dir,
                desc_prefix="RealBokeh",
            )
        dataset = RealBokehFocalDepth(
            data_path=args.data_dir,
            mode=Mode.VAL,
            annotation_file=args.annotation_file,
            device="cpu",
            depth_map_dir=args.depth_map_dir,
            depth_mode=args.depth_mode,
            depth_normalize_scale=args.depth_normalize_scale,
            inverse_depth=args.inverse_depth,
            use_augmentation=False,
            target_apertures=args.target_apertures,
            focus_map_type=args.focus_map_type,
            focus_map_gaussian_sigma=args.focus_map_gaussian_sigma,
            focus_map_gaussian_sigma_scale=args.focus_map_gaussian_sigma_scale,
        )
    else:
        if args.target_apertures != "all":
            print("--target-apertures is ignored for --dataset ebb.")
        if args.depth_map_dir == "in_depth":
            args.depth_map_dir = (
                "depth_pro_depth"
                if args.depth_estimator == "depth-pro"
                else "da3_depth"
            )
        ensure_depth_maps(
            args=args,
            device=device,
            image_paths=ebb_image_paths(args.data_dir, Mode.VAL, "original"),
            depth_dir=args.data_dir / Mode.VAL.value / args.depth_map_dir,
            desc_prefix="EBB",
        )
        dataset = EBBFocalDepth(
            data_path=args.data_dir,
            mode=Mode.VAL,
            annotation_file=args.annotation_file,
            device="cpu",
            depth_map_dir=args.depth_map_dir,
            depth_mode=args.depth_mode,
            normalize_depth_map=False,
            depth_normalize_scale=args.depth_normalize_scale,
            inverse_depth=args.inverse_depth,
            focus_map_type=args.focus_map_type,
            focus_map_gaussian_sigma=args.focus_map_gaussian_sigma,
            focus_map_gaussian_sigma_scale=args.focus_map_gaussian_sigma_scale,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model(args).to(device)
    load_weights(model, args.checkpoint, device, strict=args.strict)

    metric_names = [m.strip().lower() for m in args.val_metrics.split(",") if m.strip()]
    metric_funcs = build_metric_funcs(metric_names, args.lpips_net)
    k_values = list(range(args.k_min, args.k_max + 1))

    print(
        f"Evaluating {args.checkpoint} on {args.dataset}:"
        f"{args.data_dir / Mode.VAL.value} "
        f"with binary K search={args.k_min}..{args.k_max}"
    )
    results, rows = evaluate_k_search(
        model=model,
        loader=loader,
        device=device,
        metric_funcs=metric_funcs,
        k_values=k_values,
        k_normalizer=args.k_normalizer,
        amp=args.amp,
        save_outputs_flag=not args.no_save_outputs,
        out_dir=image_out_dir,
        image_format=args.image_format,
    )

    write_csv(args.save_csv, rows, metric_names, results)
    write_summary_csv(args.save_summary, rows, metric_names, results)
    print(
        "best-SSIM val metrics:", ", ".join(f"{k}={v:.6f}" for k, v in results.items())
    )
    print(f"Saved per-sample results to {args.save_csv}")
    print(f"Saved averaged metrics to {args.save_summary}")
    print(f"Saved best-SSIM outputs to {image_out_dir}")


if __name__ == "__main__":
    main()
