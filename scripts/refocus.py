#!/usr/bin/env python
"""Refocus one photograph: Stage I restores the all-in-focus image, then the renderer renders the requested focus.

  python scripts/refocus.py photo.jpg out/photo --focus 640 360 --strength 0.5
  python scripts/refocus.py photo.jpg out/photo --two-planes 300 400 900 200 --strength 0.8
  python scripts/refocus.py photo.jpg out/photo --tilt-shift 0.3 --strength 0.6

Writes <out_prefix>_aif.png (restored image), _depth.png (near is bright) and _bokeh.png. The photograph is resized to
a long side of at most --long-side (sides multiples of 16) and coordinates refer to the resized image (its size is
printed). --already-sharp skips Stage I for an image that is already all-in-focus."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from flashrefocus import Refocuser
from flashrefocus.image_io import image_to_tensor, resize_to_multiple, save_image
from flashrefocus.rendering import depth_edits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("out_prefix")
    focus = ap.add_mutually_exclusive_group(required=True)
    focus.add_argument("--focus", type=float, nargs=2, metavar=("X", "Y"), help="focal point")
    focus.add_argument("--two-planes", type=float, nargs=4, metavar=("XA", "YA", "XB", "YB"),
                       help="two focal points that both stay sharp")
    focus.add_argument("--tilt-shift", type=float, metavar="ROW", help="centre of the sharp band, fraction of the height")
    ap.add_argument("--strength", type=float, default=0.5, help="blur strength K in [0, 1]")
    ap.add_argument("--detail-weight", type=float, default=1.25, help="Stage I detail weight omega")
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--already-sharp", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    photo = resize_to_multiple(Image.open(a.input), a.long_side)
    print(f"working size {photo.width}x{photo.height}")
    refocuser = Refocuser.from_pretrained(detail_weight=a.detail_weight)
    scene = refocuser.prepare(image_to_tensor(photo).cuda(), deblur=not a.already_sharp)
    save_image(scene.image, f"{out}_aif.png")
    Image.fromarray((255 * (1 - scene.depth[0].cpu().numpy())).astype(np.uint8)).save(f"{out}_depth.png")

    if a.focus:
        depth, focal_depth = None, scene.focal_depth_at(*a.focus)
    elif a.two_planes:
        xa, ya, xb, yb = a.two_planes
        focal_depth = scene.focal_depth_at(xa, ya)
        depth = depth_edits.two_focal_planes(scene.depth, focal_depth, scene.focal_depth_at(xb, yb))
    else:
        _, h, w = scene.depth.shape
        depth = depth_edits.tilt_shift(h, w, a.tilt_shift, device=scene.depth.device)
        focal_depth = float(depth_edits.depth_at(depth, w / 2, a.tilt_shift * h))
    save_image(refocuser.render(scene, a.strength, focal_depth, depth), f"{out}_bokeh.png")
    print(f"wrote {out}_aif.png, {out}_depth.png, {out}_bokeh.png")


if __name__ == "__main__":
    main()
