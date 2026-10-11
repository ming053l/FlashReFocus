"""Interactive refocusing: deblur and estimate depth once per image, then render any number of focal edits.

    refocuser = Refocuser.from_pretrained()
    scene = refocuser.prepare(read_image("photo.png", "cuda"))        # Stage I + depth, once per image
    bokeh = refocuser.render(scene, strength=0.5, focal_depth=scene.focal_depth_at(x, y))   # Stage II, per edit

A Scene holds everything an edit needs, so a video can be processed as one Scene per frame, with the focal depth
D_f and the strength K chosen per frame (for example smoothed over time) and frames rendered in batches.
Images are 1 x 3 x H x W in [0, 1] on the GPU; sides should be multiples of 16."""
from dataclasses import dataclass

import torch

from . import paths
from .deblurring.deblurrer import Deblurrer
from .image_io import read_image, save_image, to_signed, to_unit_range
from .rendering.depth import DepthEstimator
from .rendering.depth_edits import depth_at
from .rendering.renderer import load_renderer


def quantize_to_8_bits(image):
    """As saving to an 8-bit PNG and reading it back. The divisor is a tensor: CUDA divides by a Python scalar via its
    reciprocal, which differs in the last bit from image_io.image_to_tensor (division on the CPU)."""
    return (image.clamp(0, 1) * 255.0).to(torch.uint8).float() / torch.tensor(255.0, device=image.device)


@dataclass
class Scene:
    """The restored all-in-focus image I_AiF (1, 3, H, W) in [0, 1] and its depth D (1, H, W), 1 = far."""
    image: torch.Tensor
    depth: torch.Tensor

    def focal_depth_at(self, x, y):
        """D_f of the focal point (x, y) in pixels."""
        return float(depth_at(self.depth, x, y))


class Refocuser:
    def __init__(self, deblurrer, depth_estimator, renderer, render_dtype=torch.bfloat16, channels_last=False):
        self.deblurrer, self.depth_estimator, self.renderer = deblurrer, depth_estimator, renderer
        self.render_dtype, self.channels_last = render_dtype, channels_last

    @classmethod
    def from_pretrained(cls, detail_weight=1.25, refiner=paths.REFINER, refiner_dtype=torch.bfloat16,
                        render_dtype=torch.bfloat16, channels_last=False, merge_vae_lora=False, deblurring=True,
                        device="cuda"):
        """The released models. Reported results use the defaults; the cost benchmark (Table 6) also uses
        refiner_dtype=float16 and channels_last=True. render_dtype=None renders in fp32. deblurring=False loads only
        the depth estimator and the renderer, for images that are already all-in-focus."""
        deblurrer = Deblurrer.from_pretrained(refiner=refiner, detail_weight=detail_weight, refiner_dtype=refiner_dtype,
                                              merge_vae_lora=merge_vae_lora) if deblurring else None
        refocuser = cls(deblurrer, DepthEstimator.from_pretrained(device=device), load_renderer(paths.RENDERER, device),
                        render_dtype, channels_last)
        if channels_last:
            modules = [refocuser.renderer]
            if deblurrer is not None:
                modules += [deblurrer.model.vae, deblurrer.model.unet] + ([deblurrer.refiner] if deblurrer.refiner is not None else [])
            for module in modules:
                module.to(memory_format=torch.channels_last)
        return refocuser

    def scene_from_file(self, path, deblur=False):
        """Scene of an image file (by default already all-in-focus)."""
        return self.prepare(read_image(path, self.renderer.mean.device), deblur=deblur)

    def render_to_file(self, scene, strength, focal_depth, path, depth=None):
        bokeh = self.render(scene, strength, focal_depth, depth)
        save_image(bokeh, path)
        return bokeh

    def _memory_format(self, x):
        return x.contiguous(memory_format=torch.channels_last) if self.channels_last else x

    @torch.no_grad()
    def deblur(self, image):
        """Stage I: blurred image -> all-in-focus image, both in [0, 1] (the output is not clamped)."""
        return to_unit_range(self.deblurrer(self._memory_format(to_signed(image))))

    @torch.no_grad()
    def prepare(self, image, deblur=True):
        """Scene of an image: Stage I (skipped for an image that is already all-in-focus) and its depth. The restored
        image is quantised to 8 bits, as when it is saved and reloaded."""
        aif = quantize_to_8_bits(self.deblur(image)) if deblur else image
        return Scene(aif, self.depth_estimator(aif))

    @torch.no_grad()
    def render(self, scene, strength, focal_depth, depth=None):
        """Stage II: bokeh (b, 3, H, W) in [0, 1] with blur strength K and focal depth D_f. `depth` replaces the
        scene depth, for depth edits (two focal planes, tilt-shift)."""
        depth = scene.depth if depth is None else depth
        image = self._memory_format(scene.image)
        device = image.device
        with torch.autocast(device.type, dtype=self.render_dtype or torch.float32, enabled=self.render_dtype is not None):
            return self.renderer(image, torch.tensor([[float(strength)]], device=device), depth[None],
                                 torch.tensor([[float(focal_depth)]], device=device)).float()
