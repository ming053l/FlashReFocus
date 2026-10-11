"""Image helpers shared by the evaluation scripts (HxWx3 float32 arrays in [0, 1])."""
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")   # needs libx264
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except OSError:
    FONT = ImageFont.load_default()


def load_rgb(path, size=None):
    """HxWx3 float32 in [0, 1]. size (w, h): GenRefocus resizes a 1024x682 input to multiples of 16 (1024x688) before
    generation, so its output is resized back to the all-in-focus size for pixel-aligned comparison."""
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def save_rgb(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def masked_psnr(a, b, mask):
    if mask.sum() < 100:
        return float("nan")
    mse = ((a - b) ** 2)[mask].mean()
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def laplacian_sharpness(img, mask):
    """Mean |Laplacian| of the grey image over mask."""
    return float(np.abs(ndimage.laplace(img.mean(-1)))[mask].mean())


def mark_point(img, x, y, r=10):
    im = Image.fromarray((img * 255).astype(np.uint8))
    ImageDraw.Draw(im).ellipse([x - r, y - r, x + r, y + r], outline=(0, 255, 0), width=3)
    return np.asarray(im).astype(np.float32) / 255.0


def add_label(img, text):
    im = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, d.textlength(text, font=FONT) + 16, 34], fill=(0, 0, 0))
    d.text((8, 5), text, fill=(255, 255, 255), font=FONT)
    return np.asarray(im).astype(np.float32) / 255.0


def encode_video(frames_dir, pattern, dst, fps):
    """MP4 (H.264) and GIF of numbered frames."""
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / pattern),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    str(dst.with_suffix(".mp4"))], check=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(dst.with_suffix(".mp4")), "-vf",
                    "fps=%d,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" % min(fps, 15),
                    str(dst.with_suffix(".gif"))], check=True)
