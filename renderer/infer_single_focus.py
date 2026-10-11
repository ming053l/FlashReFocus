"""Stage II on one image: Depth Anything V2 depth, a focal point and a blur strength -> rendered bokeh.
This is the reference implementation (depth post-processed on the CPU as for the training data); pipeline/fast_refocus.py
computes the same on the GPU for repeated edits of one image.

  python renderer/infer_single_focus.py -i aif.png -o out.png --focus-x 640 --focus-y 360 --bokeh-strength 0.5 --amp"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image, to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
from method.model import load_renderer  # noqa: E402

DA2_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Render bokeh for one all-in-focus image and one focal point.")
    ap.add_argument("--input", "-i", type=Path, required=True)
    ap.add_argument("--output", "-o", type=Path, required=True)
    ap.add_argument("--marked-output", type=Path, default=None, help="copy with the focal point marked "
                                                                      "(default <output>_focus.<ext>)")
    ap.add_argument("--focus-x", type=float, required=True)
    ap.add_argument("--focus-y", type=float, required=True)
    ap.add_argument("--bokeh-strength", type=float, required=True, help="blur strength K in [0, 1]")
    ap.add_argument("--resize", type=int, default=None, help="resize the long side (--resize-short: the short side) "
                                                             "first; the focal point is given in input coordinates")
    ap.add_argument("--resize-short", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=paths.RENDERER)
    ap.add_argument("--da2-checkpoint", type=Path, default=paths.DA2)
    ap.add_argument("--da2-encoder", default="vitl", choices=list(DA2_CONFIGS))
    ap.add_argument("--da2-input-size", type=int, default=1024)
    ap.add_argument("--mark-size", type=int, default=None)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for the renderer (used for all results)")
    return ap.parse_args(argv)


def load_source_image(path, resize=None, resize_short=False):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    if resize:
        scale = resize / (min(w, h) if resize_short else max(w, h))
        image = image.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)
    return image, (w, h)


def load_da2_model(encoder, checkpoint_path, device):
    if str(paths.DA2_ROOT) not in sys.path:
        sys.path.insert(0, str(paths.DA2_ROOT))
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(**DA2_CONFIGS[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model.to(device).eval()


def predict_da2_depth(image, da2_model, input_size, device):
    """Relative inverse depth of a PIL image (fp16 autocast, as for the cached training depth)."""
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
        depth = da2_model.infer_image(bgr, input_size)
    return depth.astype(np.float32)


def process_da2_depth(da2_depth, size_wh):
    """Resize to the image, min-max normalize and invert: D in [0, 1], 1 = far (1 x H x W)."""
    depth_img = Image.fromarray(da2_depth.astype(np.float32), mode="F")
    if depth_img.size != tuple(size_wh):
        depth_img = depth_img.resize(tuple(size_wh), Image.BILINEAR)
    d = np.nan_to_num(np.asarray(depth_img).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(d.reshape(-1), (0.0, 100.0))
    d = np.clip(d, low, high)
    d = (d - low) / (high - low) if abs(high - low) > 1e-12 else np.zeros_like(d, dtype=np.float32)
    return torch.from_numpy(1.0 - d).unsqueeze(0).float().clamp(0, 1)


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


class RenderEngine:
    """Depth Anything V2 and the renderer, loaded once."""

    def __init__(self, checkpoint=paths.RENDERER, da2_checkpoint=paths.DA2, da2_encoder="vitl", da2_input_size=1024,
                 device="cuda", amp=True):
        self.device = torch.device(device)
        self.da2_input_size = da2_input_size
        self.amp = amp and self.device.type == "cuda"
        self.da2_model = load_da2_model(da2_encoder, da2_checkpoint, self.device)
        self.model = load_renderer(checkpoint, self.device)

    def infer_image(self, input_path, output_path, focus_x, focus_y, bokeh_strength, marked_output=None,
                    resize=None, resize_short=False, mark_size=None):
        if not 0.0 <= bokeh_strength <= 1.0:
            raise ValueError("the bokeh strength must be in [0, 1]")
        source_pil, (ow, oh) = load_source_image(input_path, resize, resize_short)
        pw, ph = source_pil.size
        depth = process_da2_depth(predict_da2_depth(source_pil, self.da2_model, self.da2_input_size, self.device),
                                  source_pil.size)
        x = max(0, min(pw - 1, int(round(focus_x * pw / ow))))
        y = max(0, min(ph - 1, int(round(focus_y * ph / oh))))
        source = to_tensor(source_pil).float().clamp(0, 1).unsqueeze(0).to(self.device)
        strength = torch.tensor([[bokeh_strength]], device=self.device)
        focus = torch.tensor([[float(depth[0, y, x])]], device=self.device)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=self.amp, dtype=torch.bfloat16):
            pred = self.model(source, strength, depth.unsqueeze(0).to(self.device), focus)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = to_pil_image(pred.clamp(0, 1).squeeze(0).float().cpu())
        image.save(output_path)
        marked = Path(marked_output) if marked_output else default_marked_output_path(output_path)
        draw_focus_marker(image, (x, y), mark_size=mark_size).save(marked)
        return image


def main():
    a = parse_args()
    eng = RenderEngine(a.checkpoint, a.da2_checkpoint, a.da2_encoder, a.da2_input_size, amp=a.amp)
    eng.infer_image(a.input, a.output, a.focus_x, a.focus_y, a.bokeh_strength, a.marked_output, a.resize,
                    a.resize_short, a.mark_size)
    print(f"Saved {a.output}")


if __name__ == "__main__":
    main()
