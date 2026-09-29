import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from dataset.loader_dma import RealBokeh
from dataset.util import Mode
from method.config_dma import build_bokehlicious_dma
from train import build_metric_funcs, extract_model_state_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Bokehlicious DMA model on RealBokeh validation data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/ssddd/chingheng/Deblurring-Dataset/RealBokeh"),
        help="RealBokeh dataset root containing validation/metadata and validation/pseudo_defocus_map.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=True,
        help="Path to a DMA checkpoint, e.g. checkpoints/train_dma_small/best.pt.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "large", "defocus_deblur"],
        help="DMA model size/config.",
    )
    parser.add_argument("--binary-bokeh", action="store_true")
    parser.add_argument("--defocus-deblur", action="store_true")
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
        default="psnr,ssim,lpips",
        help="Comma-separated metrics: psnr,ssim,lpips.",
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone used by torchmetrics LPIPS.",
    )
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/evaluate_dma"),
        help="Directory for --save-outputs images.",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="Output image format when --save-outputs is set.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Load checkpoint with strict=True. Default is strict=False.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_run_size(args):
    if args.defocus_deblur:
        return "defocus_deblur"
    if args.binary_bokeh and args.size in ("small", "large"):
        return f"{args.size}_bin"
    return args.size


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


def update_tracker(tracker, metric_funcs, outputs, targets, batch_size):
    for metric_name, metric_fn in metric_funcs.items():
        if metric_name == "lpips":
            metric_value = metric_fn(
                outputs.clamp(0.0, 1.0), targets.clamp(0.0, 1.0)
            ).item()
            metric_fn.reset()
        else:
            metric_value = metric_fn(outputs, targets).item()
        tracker.update(metric_name, metric_value, n=batch_size)


def save_batch_outputs(outputs, image_names, out_dir, image_format):
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image_names, str):
        image_names = [image_names]
    for output, image_name in zip(outputs, image_names):
        out_path = out_dir / f"{Path(image_name).stem}.{image_format}"
        to_pil_image(output.detach().cpu().clamp(0, 1)).save(out_path)


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    metric_funcs,
    amp=False,
    save_outputs=False,
    out_dir=None,
    image_format="png",
):
    from train import MetricTracker

    model.eval()
    for metric_fn in metric_funcs.values():
        if isinstance(metric_fn, torch.nn.Module):
            metric_fn.to(device)
            metric_fn.eval()

    tracker = MetricTracker(*metric_funcs.keys())
    use_amp = bool(amp and device.type == "cuda")

    for batch in tqdm(loader, desc="val"):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=torch.bfloat16,
        ):
            outputs = model(
                source=batch["source"],
                bokeh_strength=batch["bokeh_strength"],
                pos_map=batch["pos_map"],
                bokeh_strength_map=batch["bokeh_strength_map"],
                depth_map=batch.get("depth_map"),
                disp_focus=batch.get("disp_focus"),
            )

        update_tracker(
            tracker=tracker,
            metric_funcs=metric_funcs,
            outputs=outputs,
            targets=batch["target"],
            batch_size=batch["source"].size(0),
        )

        if save_outputs:
            save_batch_outputs(outputs, batch["image_name"], out_dir, image_format)

    return tracker.result()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")

    device = resolve_device(args.device)
    run_size = resolve_run_size(args)

    dataset = RealBokeh(
        data_path=args.data_dir,
        mode=Mode.VAL,
        binary_bokeh=args.binary_bokeh,
        defocus_deblur_mode=args.defocus_deblur,
        device="cpu",
        use_augmentation=False,
        require_depth_map=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_bokehlicious_dma(run_size).to(device)
    load_weights(model, args.checkpoint, device, strict=args.strict)

    metric_names = [m.strip().lower() for m in args.val_metrics.split(",") if m.strip()]
    metric_funcs = build_metric_funcs(metric_names, args.lpips_net)

    print(f"Evaluating DMA {run_size} on {args.data_dir / Mode.VAL.value}")
    results = evaluate(
        model=model,
        loader=loader,
        device=device,
        metric_funcs=metric_funcs,
        amp=args.amp,
        save_outputs=args.save_outputs,
        out_dir=args.out_dir,
        image_format=args.image_format,
    )

    print("val metrics:", ", ".join(f"{k}={v:.6f}" for k, v in results.items()))
    if args.save_outputs:
        print(f"Saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
