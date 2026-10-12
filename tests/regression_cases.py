"""The outputs pinned by tests/test_regression.py: for each test image, the restored all-in-focus image, its depth,
a refocused image, two focal planes and tilt-shift. Regenerate the references only when an output changes on purpose:
    python tests/regression_cases.py"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flashrefocus.image_io import read_image, tensor_to_uint8
from flashrefocus.rendering import depth_edits

HERE = Path(__file__).resolve().parent
IMAGES = ("flower", "dock")


def outputs(refocuser, name):
    """name -> HxWx3 uint8 images and the depth as HxW uint16."""
    scene = refocuser.prepare(read_image(HERE / "data" / f"{name}.png", "cuda"))
    _, h, w = scene.depth.shape
    focal = scene.focal_depth_at(0.6 * w, 0.7 * h)
    two_planes = depth_edits.two_focal_planes(scene.depth, scene.focal_depth_at(0.3 * w, 0.6 * h),
                                              scene.focal_depth_at(0.8 * w, 0.2 * h))
    tilt = depth_edits.tilt_shift(h, w, 0.4, device="cuda")
    return {
        "aif": tensor_to_uint8(scene.image),
        "depth": np.round(scene.depth[0].cpu().numpy() * 65535).astype(np.uint16),
        "bokeh": tensor_to_uint8(refocuser.render(scene, 0.5, focal)),
        "two_planes": tensor_to_uint8(refocuser.render(scene, 0.8, scene.focal_depth_at(0.3 * w, 0.6 * h), two_planes)),
        "tilt_shift": tensor_to_uint8(refocuser.render(scene, 1.0, 0.5, tilt)),
    }


def reference_path(name, kind):
    return HERE / "reference" / f"{name}_{kind}.png"


if __name__ == "__main__":
    from flashrefocus import Refocuser
    torch.backends.cudnn.benchmark = False
    refocuser = Refocuser.from_pretrained()
    for name in IMAGES:
        for kind, array in outputs(refocuser, name).items():
            Image.fromarray(array).save(reference_path(name, kind))
            print("wrote", reference_path(name, kind))
