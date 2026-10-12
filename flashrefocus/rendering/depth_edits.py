"""Depth edits (Sec. 4.3, Appendix H): the renderer reads an explicit depth map, so replacing it changes how the image
is focused without retraining. Every function returns a depth map (1, H, W) in [0, 1] to render with."""
import torch


def depth_at(depth, x, y):
    """D at pixel (x, y), clamped to the image: the focal depth D_f of a focal point."""
    _, h, w = depth.shape
    return depth[0, max(0, min(h - 1, int(round(y)))), max(0, min(w - 1, int(round(x))))]


def two_focal_planes(depth, focal_depth_a, focal_depth_b):
    """D' = D_A + min(|D - D_A|, |D - D_B|): every pixel is blurred by its distance to the nearer of two focal planes.
    Render D' with focal depth D_A. If adding the distance would clip more pixels at 1 than subtracting it would clip at
    0 (a far plane A), it is subtracted instead: adding it to D_A = 1 would make the whole image sharp."""
    da, db = float(focal_depth_a), float(focal_depth_b)
    distance = torch.minimum((depth - da).abs(), (depth - db).abs())
    subtract = int((da - distance < 0).sum()) < int((da + distance > 1).sum())
    return (da - distance if subtract else da + distance).clamp(0, 1)


def tilt_shift(height, width, row, slope=3.0, band=0.06, device="cpu"):
    """A vertical ramp with a flat band of half-height `band` centred at `row` (fractions of the height): a horizontal
    band stays sharp and blur grows above and below it. Render it with the focal point at (W / 2, row * H)."""
    yy = torch.linspace(0, 1, height, device=device).view(height, 1).expand(height, width)
    dy = (yy - row).abs().sub(band).clamp(min=0) * torch.sign(yy - row)
    return (0.5 + slope * dy / 2).clamp(0, 1)[None]
