"""Refocusing pipeline for interactive use: restore once, then render any number of focal edits.

  s1 = FastStageI()                     # Stage I: one-step diffusion + decoder refiner (weights from paths.py);
                                        # s1.omega = ... changes the detail weight
  s1("input.png", "aif.png")            # once per photograph
  rd = FastRenderer()                   # Stage II: Depth Anything V2 + renderer
  rd.set_image("aif.png")               # depth is estimated once and cached
  rd.render(x, y, strength, "out.png")  # per edit; strength in [0, 1]

Compared with the reference scripts (stage1/deblur.py, renderer/infer_single_focus.py) only the code around
the models changes, so the outputs are the same up to floating point: the empty-prompt text embedding is computed once,
Depth Anything V2 pre- and post-processing and the focus map run on the GPU, the depth of an image is reused by every
edit, and PNGs are written with a fast zlib level. Options: cuDNN autotuning, channels_last, fp16 refiner, renderer
autocast (bf16 by default) and torch.compile."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
paths.add_to_path()

PNG_LEVEL = 1


def read_rgb(path, device):
    """RGB image -> float [0, 1] tensor 1x3xHxW on device, converted as torchvision's to_tensor (model.to_unit)."""
    from model import to_unit
    return to_unit(Image.open(path).convert("RGB")).to(device)


def write_rgb(x01, path):
    """float [0, 1] 1x3xHxW -> lossless PNG, quantised on the GPU by truncation as torchvision's to_pil_image."""
    arr = (x01[0].clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path, compress_level=PNG_LEVEL)


def setup_backend(cudnn_benchmark=True):
    torch.backends.cudnn.benchmark = cudnn_benchmark


class FastStageI:
    """Stage I for the pipeline: restore an image file into an all-in-focus PNG (stage1/model.py does the work)."""

    def __init__(self, ckpt=paths.STAGE1, refiner=paths.REFINER, lambda_sem=1.25, refiner_fp16=False, compile_=False,
                 channels_last=False, compile_mode=None, merge_vae_lora=False):
        """merge_vae_lora: faster VAE encoder, outputs ~60 dB from the exact ones (see model.load_stage1)."""
        from model import Deblurrer, load_refiner, load_stage1, stage1_params
        self.deblur = Deblurrer(load_stage1(ckpt, omega=lambda_sem, merge_vae_lora=merge_vae_lora),
                                load_refiner(refiner) if refiner is not None else None)
        self.m, self.ref = self.deblur.m, self.deblur.ref
        self.params = stage1_params(self.m, self.ref)
        if self.ref is not None and refiner_fp16:
            self.ref = self.deblur.ref = self.ref.to(torch.float16)
        if channels_last:
            for mod in (self.m.vae, self.m.unet) + ((self.ref,) if self.ref is not None else ()):
                mod.to(memory_format=torch.channels_last)
        self.channels_last = channels_last
        if compile_:
            # compile forward functions, not the modules: the omega path swaps U-Net weights by parameter name,
            # which a module wrapper would rename (_orig_mod.*)
            for mod in (self.m.unet, self.m.vae.encoder, self.ref if self.ref is not None else self.m.vae.decoder):
                mod.forward = torch.compile(mod.forward, mode=compile_mode)

    @property
    def omega(self):
        return self.deblur.omega

    @omega.setter
    def omega(self, value):
        self.deblur.omega = value

    def run_tensor(self, x):
        """[-1, 1] image tensor -> restored image tensor (not clamped)."""
        return self.deblur(x)

    def __call__(self, src, dst):
        x = read_rgb(src, "cuda") * 2 - 1
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        write_rgb(self.run_tensor(x).float() * 0.5 + 0.5, dst)


class FastRenderer:
    def __init__(self, sdpa_da2=True, channels_last=False, amp_dtype=torch.bfloat16, compile_=False, compile_mode=None):
        # bf16 autocast by default, as in the bokeh-rendering evaluation (test_focal_depth_k_search_p.py --amp);
        # amp_dtype=None renders in fp32
        from infer_single_focus import BokehliciousFocusEngine, parse_args as parse_bokeh_args
        bc = parse_bokeh_args(["--input", "a", "--output", "b", "--focus-x", "0", "--focus-y", "0", "--bokeh-strength", "0.5"])
        self.bc = bc
        eng = BokehliciousFocusEngine(checkpoint=bc.checkpoint, size=bc.size, attn_version=bc.attn_version, da2_root=bc.da2_root,
                                      da2_checkpoint=bc.da2_checkpoint, da2_encoder=bc.da2_encoder,
                                      da2_input_size=bc.da2_input_size, no_da2_amp=bc.no_da2_amp,
                                      da2_amp_dtype=bc.da2_amp_dtype, device="cuda", amp=bc.amp)
        if sdpa_da2:
            import sdpa_patch
            sdpa_patch.apply(da2=True, renderer=False)
        from depth_anything_v2.util.transform import Resize
        from dataset.util import make_point_focus_map
        self.da2, self.model, self.eng = eng.da2_model, eng.model, eng
        self.Resize, self.make_point_focus_map = Resize, make_point_focus_map
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
        self.params = sum(p.numel() for p in self.da2.parameters()) + sum(p.numel() for p in self.model.parameters())
        self.src = self.depth = None
        self.channels_last, self.amp_dtype = channels_last, amp_dtype
        if channels_last:
            self.model.to(memory_format=torch.channels_last)
        if compile_:
            self.model.forward = torch.compile(self.model.forward, mode=compile_mode)

    @torch.no_grad()
    def set_image(self, path):
        """Read the AiF image and compute its depth once (reused by every focal edit on this image)."""
        src = read_rgb(path, "cuda")
        h, w = src.shape[-2:]
        s = self.eng.da2_input_size
        tw, th = self.Resize(width=s, height=s, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                             resize_method="lower_bound").get_size(w, h)
        x = F.interpolate(src, (th, tw), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.eng.da2_amp):
            d = self.da2(x)
        d = F.interpolate(d[:, None].float(), (h, w), mode="bilinear", align_corners=True)[0]
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = d.min(), d.max()
        d = (d - lo) / (hi - lo) if float(hi - lo) > 1e-12 else torch.zeros_like(d)
        self.src, self.depth = src, (1.0 - d).clamp(0, 1)            # 1 x H x W, as process_da2_depth_like_dataset

    @torch.no_grad()
    def render(self, focus_x, focus_y, strength, dst=None, depth=None):
        """Bokeh at focal point (x, y) and blur strength; depth (1 x H x W, 1 = far) replaces the estimated depth, as
        for two focal planes or tilt-shift. Returns 1x3xHxW in [0, 1]; writes a PNG when dst is given."""
        depth = self.depth if depth is None else depth
        _, h, w = depth.shape
        x = max(0, min(w - 1, int(round(focus_x)))); y = max(0, min(h - 1, int(round(focus_y))))
        disp = depth[0, y, x].reshape(1, 1)
        fm = self.make_point_focus_map(w, h, x, y, radius=max(h, w), map_type=self.bc.focus_map_type, device="cuda")[None].float()
        src = self.src.contiguous(memory_format=torch.channels_last) if self.channels_last else self.src
        with torch.autocast("cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None):
            pred = self.model(source=src, bokeh_strength=torch.tensor([[strength]], device="cuda"), focus_map=fm,
                              depth_map=depth[None], disp_focus=disp)
        if dst is not None:
            write_rgb(pred.float(), dst)
        return pred

    def depth_at(self, x, y):
        _, h, w = self.depth.shape
        return float(self.depth[0, max(0, min(h - 1, int(y))), max(0, min(w - 1, int(x)))])

    def two_plane_depth(self, a, b):
        """Edited depth for two focal planes through points a and b: every pixel is blurred by its distance to the
        nearer plane. Render it at point a."""
        da, db = self.depth_at(*a), self.depth_at(*b)
        return (da + torch.minimum((self.depth - da).abs(), (self.depth - db).abs())).clamp(0, 1)

    def tilt_shift_depth(self, row, slope=3.0, band=0.06):
        """Edited depth for tilt-shift: a vertical ramp with a flat band of half-height `band` centred at `row`
        (fractions of the height). Render it at (W / 2, row * H)."""
        _, h, w = self.depth.shape
        yy = torch.linspace(0, 1, h, device=self.depth.device).view(h, 1).expand(h, w)
        dy = (yy - row).abs().sub(band).clamp(min=0) * torch.sign(yy - row)
        return (0.5 + slope * dy / 2).clamp(0, 1)[None]
