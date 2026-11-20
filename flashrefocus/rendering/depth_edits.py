"""Depth edits (Sec. 4.3 and the appendix): the renderer reads an explicit depth map, so replacing it changes how the image
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


def _grid(height, width, device):
    """Row and column coordinates in [0, 1], each H x W."""
    yy = torch.linspace(0, 1, height, device=device).view(height, 1).expand(height, width)
    xx = torch.linspace(0, 1, width, device=device).view(1, width).expand(height, width)
    return yy, xx


def _ramp(signed_distance, band, slope):
    """0.5 within `band` of zero, growing (or falling, for negative distances) linearly beyond it."""
    d = signed_distance.abs().sub(band).clamp(min=0) * torch.sign(signed_distance)
    return (0.5 + slope * d / 2).clamp(0, 1)[None]


def focus_range(depth, focal_depth, width):
    """A thick focal plane: D' = D_f + sign(D - D_f) max(|D - D_f| - width, 0). Every depth within `width` of D_f stays
    sharp, and the rest keeps its distance minus `width`. Render with the same D_f."""
    s = depth - float(focal_depth)
    return (float(focal_depth) + torch.sign(s) * (s.abs() - width).clamp(min=0)).clamp(0, 1)


def radial_focus(height, width, center_x, center_y, radius=0.12, slope=2.2, device="cpu"):
    """Lensbaby-style focus: a sharp disc of `radius` (fraction of the height) around (center_x, center_y) (fractions of
    the width and height), with blur growing outward. Render with focal depth 0.5."""
    yy, xx = _grid(height, width, device)
    dist = torch.sqrt(((xx - center_x) * width / height) ** 2 + (yy - center_y) ** 2)
    return _ramp(dist, radius, slope)


def tilted_band(height, width, center_x=0.5, center_y=0.5, angle=0.0, band=0.06, slope=3.0, device="cpu"):
    """Tilt-shift at any angle: a sharp band of half-width `band` (fraction of the height) through (center_x, center_y),
    rotated by `angle` degrees (0 = horizontal), with blur growing on both sides. Render with focal depth 0.5;
    angle 0 gives the same map as tilt_shift."""
    import math
    yy, xx = _grid(height, width, device)
    t = math.radians(angle)
    signed = -(xx - center_x) * width / height * math.sin(t) + (yy - center_y) * math.cos(t)
    return _ramp(signed, band, slope)


def one_sided(depth, focal_depth, keep="foreground"):
    """Blur only one side of the focal plane. keep="foreground" clamps everything nearer than D_f onto the focal plane
    (only the background blurs); keep="background" clamps everything farther (only the foreground blurs)."""
    df = torch.tensor(float(focal_depth), device=depth.device)
    return torch.maximum(depth, df) if keep == "foreground" else torch.minimum(depth, df)


def focal_planes(depth, focal_depths):
    """Several focal planes: every pixel is blurred by its distance to the nearest of `focal_depths`. Render with the
    first focal depth. Generalises two_focal_planes (same rule for choosing the side of the first plane)."""
    d0 = float(focal_depths[0])
    distance = torch.stack([(depth - float(d)).abs() for d in focal_depths]).min(0).values
    subtract = int((d0 - distance < 0).sum()) < int((d0 + distance > 1).sum())
    return (d0 - distance if subtract else d0 + distance).clamp(0, 1)


def keep_sharp(depth, mask, focal_depth):
    """Place the masked pixels (mask: H x W or 1 x H x W, bool) on the focal plane, so they stay sharp."""
    return torch.where(mask.view_as(depth), torch.full_like(depth, float(focal_depth)), depth)


def set_mask_depth(depth, mask, value):
    """Move the masked pixels to depth `value` (0 = near, 1 = far): for example 1.0 pushes a photobomber into the far
    background, so he melts into the bokeh while the subject stays sharp. No inpainting is involved."""
    return torch.where(mask.view_as(depth), torch.full_like(depth, float(value)), depth)


def move_away(depth, mask, focal_depth, distance):
    """Move the masked pixels at least `distance` away from the focal plane, on their own side (or the side with more
    room): optical privacy blur for faces, or extra blur for one layer."""
    df = float(focal_depth)
    s = depth - df
    side = torch.where(s.abs() > 1e-3, torch.sign(s), torch.full_like(s, 1.0 if df < 0.5 else -1.0))
    moved = (df + side * torch.maximum(s.abs(), torch.full_like(s, distance))).clamp(0, 1)
    return torch.where(mask.view_as(depth), moved, depth)


def isolate(depth, mask, focal_depth, gap):
    """Keep the masked pixels on the focal plane and move every other pixel at least `gap` away from it, so everything
    around the subject blurs strongly."""
    return keep_sharp(move_away(depth, ~mask.view_as(depth), focal_depth, gap), mask, focal_depth)


def grow_mask(mask, depth, focal_depth, others=None, radius=10, tolerance=0.05):
    """Add to a subject mask the pixels next to it (within `radius` pixels) that sit near the focal depth and belong to
    no other person (`others`, optional bool mask): segmentation often misses fingers, hair and motion-blurred hands,
    which would otherwise be pushed out of focus. Only grown pixels connected to the mask are kept; holes are filled."""
    import numpy as np
    from scipy import ndimage
    m = mask.view(depth.shape[-2:]).cpu().numpy().astype(bool)
    if not m.any():
        return mask
    d = depth.view(depth.shape[-2:]).float().cpu().numpy()
    ring = ndimage.binary_dilation(m, iterations=int(radius)) & ~m
    if others is not None:
        ring &= ~others.view(depth.shape[-2:]).cpu().numpy().astype(bool)
    grown = m | (ring & (np.abs(d - float(focal_depth)) < tolerance))
    lab, _ = ndimage.label(grown)
    keep = np.unique(lab[m])
    grown = ndimage.binary_fill_holes(np.isin(lab, keep[keep > 0]))
    return torch.from_numpy(grown).to(mask.device).view_as(mask)


def spotlight(depth, mask, focal_depth, gap=0.6, others=None, grow=10):
    """Spotlight on a subject (for example the lead singer in a concert video): the subject mask, grown by `grow`
    pixels into its missed hands and hair (grow_mask), goes on the focal plane and every other pixel moves at least
    `gap` away from it, so the people around the subject blur strongly. Returns the edited depth and the grown mask."""
    if grow > 0:
        mask = grow_mask(mask, depth, focal_depth, others, radius=grow)
    return isolate(depth, mask, focal_depth, gap), mask
