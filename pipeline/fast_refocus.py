"""Faster implementation of the same FlashReFocus refocusing pipeline (same models, same outputs up to floating point).

Only the code around the models changes; every call still reads its input image from disk and writes its output as
PNG, as in the Table 6 protocol:
  * Stage I: the empty-prompt text embedding is computed once; optional fp16 refiner and torch.compile.
  * Stage II: Depth Anything V2 pre-/post-processing on the GPU (bicubic resize to DA2's own target size, ImageNet
    normalisation, min-max depth normalisation, as infer_single_focus.py does on the CPU), focus map built on the GPU,
    the depth of an AiF image is computed once and reused for every further focal edit, and no extra marker image.
  * PNG written losslessly with a fast zlib level.
Options: cudnn autotuning for the fixed input size, channels_last for the convolutional nets (VAE, refiner, renderer),
fp16 refiner, renderer autocast, and torch.compile (optionally with CUDA graphs).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEFOCUS = Path("/ssd1/chingheng/Defocus")
sys.path.insert(0, str(DEFOCUS / "Stage1_PiSA"))
sys.path.insert(0, str(DEFOCUS / "SOTA" / "Bokehlicious"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PNG_LEVEL = 1


def read_rgb(path, device):
    """uint8 PNG -> float [0, 1] tensor 1x3xHxW on device (decode on CPU, convert on GPU)."""
    arr = np.asarray(Image.open(path).convert("RGB"))
    return torch.from_numpy(arr).to(device, non_blocking=True).permute(2, 0, 1)[None].float() / 255.0


def write_rgb(x01, path):
    """float [0, 1] 1x3xHxW -> lossless PNG, quantised on the GPU by truncation as torchvision's to_pil_image."""
    arr = (x01[0].clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path, compress_level=PNG_LEVEL)


def setup_backend(cudnn_benchmark=True):
    torch.backends.cudnn.benchmark = cudnn_benchmark


class FastStageI:
    def __init__(self, ckpt, refiner, lambda_sem=1.25, refiner_fp16=False, compile_=False, channels_last=False,
                 compile_mode=None):
        import argparse
        import cost_pisa
        ns = argparse.Namespace(sd21=str(DEFOCUS / "stable-diffusion-2-1-base"), sd15=str(DEFOCUS / "stable-diffusion-v1-5"),
                                myautorefocus=str(DEFOCUS / "MyAutoRefocus"), ckpt=ckpt, refiner=refiner)
        self.m, self.ref, self.default = cost_pisa.build(ns, lambda_sem, refiner is not None, "cuda", torch.float16)
        self.params = cost_pisa.params(self.m, self.ref)
        self.wd = self.m.weight_dtype
        with torch.no_grad():
            self.emb = self.m.encode_prompt([""]).to(dtype=self.wd)
        if self.ref is not None and refiner_fp16:
            self.ref = self.ref.to(torch.float16)
        if channels_last:
            self.m.vae.to(memory_format=torch.channels_last)
            self.m.unet.to(memory_format=torch.channels_last)
            if self.ref is not None:
                self.ref.to(memory_format=torch.channels_last)
        self.channels_last = channels_last
        if compile_:
            # compile forward functions, not the modules: PiSA-SR's omega path swaps U-Net weights by parameter name,
            # which a module wrapper would rename (_orig_mod.*)
            for mod in (self.m.unet, self.m.vae.encoder, self.ref if self.ref is not None else self.m.vae.decoder):
                mod.forward = torch.compile(mod.forward, mode=compile_mode)

    @torch.no_grad()
    def run_tensor(self, x):
        m = self.m
        sf = m.vae.config.scaling_factor
        z = m.vae.encode(x.to(self.wd)).latent_dist.mode() * sf
        z_out = z - m._process_latents(z, self.emb, self.default)
        if self.ref is None:
            return m.vae.decode(z_out / sf).sample
        rd = next(self.ref.parameters()).dtype
        return self.ref(x.to(rd), latent=(z_out / sf).to(rd), sample_posterior=False)

    def __call__(self, src, dst):
        x = read_rgb(src, "cuda") * 2 - 1
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        y = self.run_tensor(x)
        write_rgb(y.float() * 0.5 + 0.5, dst)


class FastRenderer:
    def __init__(self, sdpa_da2=True, channels_last=False, amp_dtype=torch.bfloat16, compile_=False, compile_mode=None):
        # bf16 autocast by default, as in the evaluation behind the bokeh tables (test_focal_depth_k_search_p.py --amp);
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
    def render(self, focus_x, focus_y, strength, dst):
        _, h, w = self.depth.shape
        x = max(0, min(w - 1, int(round(focus_x)))); y = max(0, min(h - 1, int(round(focus_y))))
        disp = self.depth[0, y, x].reshape(1, 1)
        fm = self.make_point_focus_map(w, h, x, y, radius=max(h, w), map_type=self.bc.focus_map_type, device="cuda")[None].float()
        src = self.src.contiguous(memory_format=torch.channels_last) if self.channels_last else self.src
        with torch.autocast("cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None):
            pred = self.model(source=src, bokeh_strength=torch.tensor([[strength]], device="cuda"), focus_map=fm,
                              depth_map=self.depth[None], disp_focus=disp)
        if dst is not None:
            write_rgb(pred.float(), dst)
        return pred
