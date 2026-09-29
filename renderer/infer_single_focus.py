import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image, to_tensor

from dataset.util import make_point_focus_map
from method.config import bokehlicious_size_builder
from method.model_dma_p import BokehliciousDMA

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Bokehlicious DMA-p inference on one image with one focus point."
    )
    parser.add_argument("--input", "--img-path", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--marked-output",
        type=Path,
        default=None,
        help="Output path for the image with focus marker. Defaults to <output>_focus.<ext>.",
    )
    parser.add_argument("--focus-x", type=float, required=True)
    parser.add_argument("--focus-y", type=float, required=True)
    parser.add_argument(
        "--bokeh-embedding",
        "--bokeh-strength",
        type=float,
        required=True,
        help="Bokeh embedding/strength in [0, 1].",
    )
    parser.add_argument(
        "--resize-size",
        "--resize",
        type=int,
        default=None,
        help="Optional resize size. Long side is resized to this value by default.",
    )
    parser.add_argument(
        "--resize-short",
        action="store_true",
        help="When set, resize the short side to --resize-size instead of the long side.",
    )
    parser.add_argument(
        "--coords-after-resize",
        action="store_true",
        help="Interpret --focus-x/--focus-y in resized image coordinates.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/train_realbokeh_da2_disparity_p/best.pt",
        help="Bokehlicious DMA-p checkpoint.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "large", "defocus_deblur"],
        help="Bokehlicious model size.",
    )
    parser.add_argument(
        "--attn-version",
        type=str,
        default="defocus",
        help="DMA attention version used by the checkpoint.",
    )
    parser.add_argument(
        "--da2-root",
        type=Path,
        default=Path(
            os.environ.get("DA2_ROOT", PROJECT_ROOT.parent / "Depth-Anything-V2")
        ),
        help="Depth-Anything-V2 repository root.",
    )
    parser.add_argument(
        "--da2-checkpoint",
        type=Path,
        default=None,
        help="Depth-Anything-V2 checkpoint. Defaults to <da2-root>/checkpoints/depth_anything_v2_<encoder>.pth.",
    )
    parser.add_argument(
        "--da2-encoder",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl", "vitg"],
    )
    parser.add_argument("--da2-input-size", type=int, default=1024)
    parser.add_argument(
        "--no-da2-amp",
        action="store_true",
        help="Disable autocast for Depth-Anything-V2 inference.",
    )
    parser.add_argument(
        "--da2-amp-dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Autocast dtype for Depth-Anything-V2 on CUDA.",
    )
    parser.add_argument(
        "--focus-map-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "cosine"],
    )
    parser.add_argument(
        "--mark-size",
        type=int,
        default=None,
        help="Focus marker box size in pixels. Defaults to a size based on image dimensions.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resized_size(width, height, resize_size, resize_short):
    if resize_size is None:
        return width, height
    if resize_size <= 0:
        raise ValueError("--resize-size must be positive.")

    if resize_short:
        scale = resize_size / min(width, height)
    else:
        scale = resize_size / max(width, height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return new_w, new_h


def load_source_image(path, resize_size=None, resize_short=False):
    image = Image.open(path).convert("RGB")
    original_size = image.size
    new_size = resized_size(*original_size, resize_size, resize_short)
    if new_size != original_size:
        image = image.resize(new_size, Image.BILINEAR)
    return image, original_size


def scale_focus_xy(x, y, original_size, processed_size, coords_after_resize):
    if coords_after_resize or original_size == processed_size:
        return x, y
    ow, oh = original_size
    pw, ph = processed_size
    return x * (pw / ow), y * (ph / oh)


def clamp_point(x, y, width, height):
    x = int(round(x))
    y = int(round(y))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def add_da2_to_path(da2_root):
    da2_root = da2_root.expanduser().resolve()
    if not da2_root.is_dir():
        raise FileNotFoundError(f"Depth-Anything-V2 root not found: {da2_root}")
    sys.path.insert(0, str(da2_root))


def load_da2_model(encoder, checkpoint_path, device):
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model.to(device).eval()


def autocast_dtype(dtype_name):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported autocast dtype: {dtype_name}")


def predict_da2_depth_from_pil(
    image,
    da2_model,
    input_size,
    device,
    use_amp=True,
    amp_dtype=torch.float16,
):
    import cv2

    rgb = np.asarray(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    with torch.amp.autocast(
        device_type=device.type,
        enabled=use_amp and device.type == "cuda",
        dtype=amp_dtype,
    ):
        depth = da2_model.infer_image(bgr, input_size)
    return depth.astype(np.float32)


def process_da2_depth_like_dataset(da2_depth, size_wh, clip_percentiles=(0.0, 100.0)):
    width, height = size_wh
    depth_img = Image.fromarray(da2_depth.astype(np.float32), mode="F")
    if depth_img.size != (width, height):
        depth_img = depth_img.resize((width, height), Image.BILINEAR)

    depth_np = np.asarray(depth_img).astype(np.float32)
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)

    low, high = np.percentile(depth_np.reshape(-1), clip_percentiles)
    depth_np = np.clip(depth_np, low, high)
    denom = high - low
    if abs(denom) > 1e-12:
        depth_np = (depth_np - low) / denom
    else:
        depth_np = np.zeros_like(depth_np, dtype=np.float32)

    depth_np = 1.0 - depth_np
    return torch.from_numpy(depth_np).unsqueeze(0).float().clamp(0, 1)


def extract_model_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must be a dict or a model state_dict.")
    for key in ("model", "state_dict"):
        value = state.get(key)
        if isinstance(value, dict):
            return value
    return state


def load_bokeh_model(checkpoint_path, size, attn_version, device):
    config = bokehlicious_size_builder(size)
    config["attn_version"] = attn_version
    config["use_radiance_visualization"] = False
    model = BokehliciousDMA(**config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_model_state_dict(checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }

    model_state = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    incompatible = model.load_state_dict(filtered, strict=False)
    print(
        "Loaded bokeh checkpoint "
        f"(missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}, "
        f"used={len(filtered)}/{len(state_dict)})"
    )
    return model.eval()


def build_focus_inputs(depth_cpu, focus_x, focus_y, map_type, device, dtype):
    _, height, width = depth_cpu.shape
    x, y = clamp_point(focus_x, focus_y, width, height)
    disp_focus_value = float(depth_cpu[0, y, x].item())
    focus_map = make_point_focus_map(
        width,
        height,
        x,
        y,
        radius=max(height, width),
        map_type=map_type,
    ).float()
    disp_focus = torch.tensor([[disp_focus_value]], device=device, dtype=dtype)
    return focus_map.unsqueeze(0).to(device), disp_focus, disp_focus_value, (x, y)


def default_marked_output_path(output_path):
    return output_path.with_name(f"{output_path.stem}_focus{output_path.suffix}")


def draw_focus_marker(image, xy, mark_size=None, color=(0, 255, 0)):
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    width, height = marked.size
    x, y = xy
    if mark_size is None:
        mark_size = max(22, min(width, height) // 8)
    mark_size = max(6, int(mark_size))
    half = mark_size // 2
    line_width = max(1, round(mark_size / 18))
    segment = max(3, round(mark_size * 0.32))

    x0 = max(0, x - half)
    x1 = min(width - 1, x + half)
    y0 = max(0, y - half)
    y1 = min(height - 1, y + half)

    lines = [
        ((x0, y0), (min(x0 + segment, x1), y0)),
        ((x0, y0), (x0, min(y0 + segment, y1))),
        ((x1, y0), (max(x1 - segment, x0), y0)),
        ((x1, y0), (x1, min(y0 + segment, y1))),
        ((x0, y1), (min(x0 + segment, x1), y1)),
        ((x0, y1), (x0, max(y1 - segment, y0))),
        ((x1, y1), (max(x1 - segment, x0), y1)),
        ((x1, y1), (x1, max(y1 - segment, y0))),
    ]
    for start, end in lines:
        draw.line([start, end], fill=color, width=line_width)

    dot_radius = max(2, line_width + 1)
    draw.ellipse(
        [
            max(0, x - dot_radius),
            max(0, y - dot_radius),
            min(width - 1, x + dot_radius),
            min(height - 1, y + dot_radius),
        ],
        fill=color,
    )
    return marked


class BokehliciousFocusEngine:
    def __init__(
        self,
        checkpoint,
        size="small",
        attn_version="rel_then_defocus",
        da2_root=PROJECT_ROOT.parent / "Depth-Anything-V2",
        da2_checkpoint=None,
        da2_encoder="vitl",
        da2_input_size=1024,
        no_da2_amp=False,
        da2_amp_dtype="fp16",
        device=None,
        amp=False,
    ):
        self.device = resolve_device(device)
        self.da2_input_size = da2_input_size
        self.da2_amp = not no_da2_amp
        self.da2_amp_dtype = autocast_dtype(da2_amp_dtype)
        self.amp = amp

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Bokeh checkpoint not found: {checkpoint}")
        da2_root = Path(da2_root)
        if da2_checkpoint is None:
            da2_checkpoint = (
                da2_root / "checkpoints" / f"depth_anything_v2_{da2_encoder}.pth"
            )
        da2_checkpoint = Path(da2_checkpoint)
        if not da2_checkpoint.is_file():
            raise FileNotFoundError(
                f"Depth-Anything-V2 checkpoint not found: {da2_checkpoint}"
            )

        add_da2_to_path(da2_root)
        self.da2_model = load_da2_model(da2_encoder, da2_checkpoint, self.device)
        self.model = load_bokeh_model(checkpoint, size, attn_version, self.device)

    def infer_image(
        self,
        input_path,
        output_path,
        focus_x,
        focus_y,
        bokeh_embedding,
        marked_output=None,
        resize_size=None,
        resize_short=False,
        coords_after_resize=False,
        focus_map_type="gaussian",
        mark_size=None,
        return_diagnostics=False,
    ):
        if not 0.0 <= bokeh_embedding <= 1.0:
            raise ValueError("--bokeh-embedding must be in [0, 1].")
        input_path = Path(input_path)
        output_path = Path(output_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        source_pil, original_size = load_source_image(
            input_path,
            resize_size=resize_size,
            resize_short=resize_short,
        )
        processed_size = source_pil.size
        focus_x, focus_y = scale_focus_xy(
            focus_x,
            focus_y,
            original_size,
            processed_size,
            coords_after_resize,
        )

        da2_depth = predict_da2_depth_from_pil(
            source_pil,
            self.da2_model,
            self.da2_input_size,
            device=self.device,
            use_amp=self.da2_amp,
            amp_dtype=self.da2_amp_dtype,
        )
        depth_cpu = process_da2_depth_like_dataset(da2_depth, source_pil.size)

        source_cpu = to_tensor(source_pil).float().clamp(0, 1)
        source = source_cpu.unsqueeze(0).to(self.device)
        depth_map = depth_cpu.unsqueeze(0).to(self.device)
        bokeh_strength = torch.tensor(
            [[bokeh_embedding]],
            device=self.device,
            dtype=source.dtype,
        )
        focus_map, disp_focus, disp_focus_value, xy = build_focus_inputs(
            depth_cpu,
            focus_x,
            focus_y,
            focus_map_type,
            self.device,
            source.dtype,
        )

        use_amp = self.amp and self.device.type == "cuda"
        with torch.inference_mode():
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=use_amp,
                dtype=torch.bfloat16,
            ):
                pred = self.model(
                    source=source,
                    bokeh_strength=bokeh_strength,
                    focus_map=focus_map,
                    depth_map=depth_map,
                    disp_focus=disp_focus,
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_image = to_pil_image(pred.clamp(0, 1).squeeze(0).detach().cpu())
        output_image.save(output_path)
        marked_output = (
            Path(marked_output)
            if marked_output
            else default_marked_output_path(output_path)
        )
        marked_output.parent.mkdir(parents=True, exist_ok=True)
        marked_image = draw_focus_marker(output_image, xy, mark_size=mark_size)
        marked_image.save(marked_output)

        print(f"Device: {self.device}")
        print(f"Input: {input_path}")
        print(f"Original size: {original_size[0]}x{original_size[1]}")
        print(f"Processed size: {processed_size[0]}x{processed_size[1]}")
        print(f"Focus xy: {xy}, disp_focus={disp_focus_value:.6f}")
        print(f"Bokeh embedding: {bokeh_embedding:.6f}")
        print(f"Saved: {output_path}")
        print(f"Saved marked: {marked_output}")
        if return_diagnostics:
            return output_image, {
                "depth": depth_cpu,
                "focal_abs": (depth_cpu - disp_focus_value).abs().clamp(0.0, 1.0),
                "focal_distance": disp_focus_value,
                "focus_xy": xy,
            }
        return output_image


def main():
    args = parse_args()

    if not 0.0 <= args.bokeh_embedding <= 1.0:
        raise ValueError("--bokeh-embedding must be in [0, 1].")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Bokeh checkpoint not found: {args.checkpoint}")

    da2_checkpoint = args.da2_checkpoint
    if da2_checkpoint is None:
        da2_checkpoint = (
            args.da2_root / "checkpoints" / f"depth_anything_v2_{args.da2_encoder}.pth"
        )
    if not da2_checkpoint.is_file():
        raise FileNotFoundError(
            f"Depth-Anything-V2 checkpoint not found: {da2_checkpoint}"
        )

    device = resolve_device(args.device)
    source_pil, original_size = load_source_image(
        args.input,
        resize_size=args.resize_size,
        resize_short=args.resize_short,
    )
    processed_size = source_pil.size
    focus_x, focus_y = scale_focus_xy(
        args.focus_x,
        args.focus_y,
        original_size,
        processed_size,
        args.coords_after_resize,
    )

    add_da2_to_path(args.da2_root)
    da2_model = load_da2_model(args.da2_encoder, da2_checkpoint, device)
    da2_depth = predict_da2_depth_from_pil(
        source_pil,
        da2_model,
        args.da2_input_size,
        device=device,
        use_amp=not args.no_da2_amp,
        amp_dtype=autocast_dtype(args.da2_amp_dtype),
    )
    depth_cpu = process_da2_depth_like_dataset(da2_depth, source_pil.size)

    source_cpu = to_tensor(source_pil).float().clamp(0, 1)
    source = source_cpu.unsqueeze(0).to(device)
    depth_map = depth_cpu.unsqueeze(0).to(device)
    bokeh_strength = torch.tensor(
        [[args.bokeh_embedding]],
        device=device,
        dtype=source.dtype,
    )
    focus_map, disp_focus, disp_focus_value, xy = build_focus_inputs(
        depth_cpu,
        focus_x,
        focus_y,
        args.focus_map_type,
        device,
        source.dtype,
    )

    model = load_bokeh_model(args.checkpoint, args.size, args.attn_version, device)
    use_amp = args.amp and device.type == "cuda"
    with torch.inference_mode():
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=torch.bfloat16,
        ):
            pred = model(
                source=source,
                bokeh_strength=bokeh_strength,
                focus_map=focus_map,
                depth_map=depth_map,
                disp_focus=disp_focus,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_image = to_pil_image(pred.clamp(0, 1).squeeze(0).detach().cpu())
    output_image.save(args.output)
    marked_output = args.marked_output or default_marked_output_path(args.output)
    marked_output.parent.mkdir(parents=True, exist_ok=True)
    marked_image = draw_focus_marker(output_image, xy, mark_size=args.mark_size)
    marked_image.save(marked_output)

    print(f"Device: {device}")
    print(f"Input: {args.input}")
    print(f"Original size: {original_size[0]}x{original_size[1]}")
    print(f"Processed size: {processed_size[0]}x{processed_size[1]}")
    print(f"Focus xy: {xy}, disp_focus={disp_focus_value:.6f}")
    print(f"Bokeh embedding: {args.bokeh_embedding:.6f}")
    print(f"Saved: {args.output}")
    print(f"Saved marked: {marked_output}")


if __name__ == "__main__":
    main()
