"""Image <-> tensor conversion used everywhere, so that every script feeds the models the same values.

Images are 1 x 3 x H x W float tensors in [0, 1]; Stage I works in [-1, 1] (to_signed / to_unit_range). Conversion
matches torchvision's to_tensor / to_pil_image: division by 255, and quantisation by truncation."""
import numpy as np
import torch
from PIL import Image


def image_to_tensor(image, multiple=1):
    """PIL image (or HxWx3 uint8 array) -> 1x3xHxW float in [0, 1] on the CPU, cropped (bottom/right) to a multiple."""
    arr = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    h, w = arr.shape[0] - arr.shape[0] % multiple, arr.shape[1] - arr.shape[1] % multiple
    return torch.from_numpy(np.ascontiguousarray(arr[:h, :w])).permute(2, 0, 1).contiguous().float().div(255)[None]


def read_image(path, device="cpu", multiple=1):
    return image_to_tensor(Image.open(path), multiple).to(device)


def tensor_to_uint8(x):
    """1x3xHxW float in [0, 1] -> HxWx3 uint8, quantised by truncation (quantised on the tensor's device)."""
    return (x[0].float().clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def save_image(x, path, compress_level=1):
    """Lossless PNG of a 1x3xHxW float tensor in [0, 1]; a low zlib level keeps writing fast."""
    Image.fromarray(tensor_to_uint8(x)).save(path, compress_level=compress_level)


def to_signed(x):
    """[0, 1] -> [-1, 1] (Stage I input range)."""
    return x * 2 - 1


def to_unit_range(y):
    """[-1, 1] -> [0, 1] in float32, not clamped."""
    return y.float() * 0.5 + 0.5


def resize_to_multiple(image, long_side, multiple=16):
    """PIL image with its long side at most `long_side` and both sides multiples of `multiple` (LANCZOS)."""
    s = min(1.0, long_side / max(image.size))
    w = max(multiple, int(image.width * s) // multiple * multiple)
    h = max(multiple, int(image.height * s) // multiple * multiple)
    return image.convert("RGB").resize((w, h), Image.LANCZOS)
