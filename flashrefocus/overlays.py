"""Burnt-in overlays (titles, captions, logos) in refocused video.

The renderer would blur a title together with the scene, and as the focus changes from frame to frame the blurred
letters glow and flicker. Fill the overlay from its surroundings before depth and rendering (fill_region), and paste
the sharp overlay back from the input afterwards (paste_back)."""
import torch
import torch.nn.functional as F


def fill_region(image, mask, passes=4, kernel=15):
    """image (b, 3, h, w) in [0, 1]; mask (b or 1, 1, h, w) bool or float (1 = overlay). The masked pixels are replaced
    by a normalized convolution of the unmasked ones at half resolution (the overlay disappears into its background)."""
    m = mask.to(image.dtype)
    m = (F.max_pool2d(m, 7, 1, 3) > 0.02).to(image.dtype)          # a little wider than the overlay's anti-aliasing
    keep = 1 - m
    xs, ks = F.avg_pool2d(image * keep, 2), F.avg_pool2d(keep, 2)
    for _ in range(passes):
        xs = F.avg_pool2d(xs, kernel, 1, kernel // 2, count_include_pad=False)
        ks = F.avg_pool2d(ks, kernel, 1, kernel // 2, count_include_pad=False)
    fill = F.interpolate(xs / ks.clamp_min(1e-4), size=image.shape[-2:], mode="bilinear", align_corners=False)
    return image * keep + fill * (1 - keep)


def paste_back(rendered, original, mask, feather=3):
    """Put the overlay pixels of `original` back onto `rendered` (both (b, 3, h, w)), with a soft edge."""
    a = mask.to(rendered.dtype)
    if feather > 0:
        a = F.avg_pool2d(a, 2 * feather + 1, 1, feather, count_include_pad=False)
    return rendered * (1 - a) + original * a
