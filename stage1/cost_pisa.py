#!/usr/bin/env python
"""Deblurring cost of the PiSA-SR Stage I at 1680x1120 (Table 2 protocol).

Configurations: plain (one UNet pass, SD2.1 VAE decoder), +refiner (the DeReFocus VAE refiner decodes the latent),
+refiner with lambda_sem = 1.25 (PiSA-SR adjustable mode: two UNet passes). Inference is exactly as in
run_pisa_refiner.py: text encoder (empty prompt), VAE encoder, UNet at t=1, then the decoder or refiner (the refiner
runs its own encoder pass on the input, as implemented).

  --flops : FLOPs of one traced run with torch.utils.flop_counter (one multiply-add = 2 FLOPs, element-wise operations
            excluded), as refocus_bench/flops_deblur.py; counts depend only on tensor shapes, so the GPU run is exact.
  --time  : mean time over --runs after --warmup warm-up runs and peak allocated memory on the current GPU,
            excluding disk I/O (Table 2 protocol).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
H, W = 1120, 1680


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def build(a, lambda_sem, refiner, device, dtype):
    from run_pisa_zeroshot import PiSADeterministic
    default = lambda_sem == 1.0
    args = argparse.Namespace(pretrained_model_path=a.sd21, pretrained_path=a.ckpt, lambda_pix=1.0, lambda_sem=lambda_sem,
                              mixed_precision="fp16" if dtype == torch.float16 else "fp32", default=default,
                              vae_decoder_tiled_size=100000, vae_encoder_tiled_size=100000, latent_tiled_size=100000,
                              latent_tiled_overlap=0, seed=42)
    m = PiSADeterministic(args)
    m.set_eval()
    ref = None
    if refiner:
        from diffusers import AutoencoderKL
        sys.path.insert(0, a.myautorefocus)
        from refiner.vae_refiner_wo_d import VAEMultiScaleFusionDecoder
        ref = VAEMultiScaleFusionDecoder(AutoencoderKL.from_pretrained(a.sd15, subfolder="vae"), depth=2,
                                         map_hidden_channels=32, use_pixel_weighting=False, use_high_freq_info=True)
        ref.load_state_dict(torch.load(a.refiner, map_location="cpu"), strict=False)
        ref = ref.to(device, dtype=torch.bfloat16).eval()   # the refiner was trained in bf16
    return m, ref, default


def params(m, ref):
    """Every model the configuration loads (text encoder, UNet with merged LoRAs, VAE; plus the refiner)."""
    p = n_params(m.text_encoder) + n_params(m.unet) + n_params(m.vae)
    if ref is not None:
        p += sum(v.numel() for k, v in ref.state_dict().items() if not k.startswith("vae.encoder") and not k.startswith("vae.quant_conv"))
    return p


@torch.no_grad()
def run(m, ref, default, x):
    wd = m.weight_dtype
    emb = m.encode_prompt([""]).to(dtype=wd)
    sf = m.vae.config.scaling_factor
    z = m.vae.encode(x.to(wd)).latent_dist.mode() * sf
    z_out = z - m._process_latents(z, emb, default)
    if ref is None:
        return m.vae.decode(z_out / sf).sample
    rd = next(ref.parameters()).dtype
    return ref(x.to(rd), latent=(z_out / sf).to(rd), sample_posterior=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["plain", "refiner", "refiner_l125"], required=True)
    ap.add_argument("--flops", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--refiner", default=None)
    ap.add_argument("--sd21", default="/ssd1/chingheng/Defocus/stable-diffusion-2-1-base")
    ap.add_argument("--sd15", default="/ssd1/chingheng/Defocus/stable-diffusion-v1-5")
    ap.add_argument("--myautorefocus", default="/ssd1/chingheng/Defocus/MyAutoRefocus")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--height", type=int, default=H)
    ap.add_argument("--width", type=int, default=W)
    a = ap.parse_args()
    globals().update(H=a.height, W=a.width)
    lam = 1.25 if a.config == "refiner_l125" else 1.0
    use_ref = a.config != "plain"
    res = {"config": a.config, "resolution": f"{W}x{H}"}
    m, ref, default = build(a, lam, use_ref, "cuda", torch.float16)
    x = torch.rand(1, 3, H, W, device="cuda") * 2 - 1
    res["params_M"] = params(m, ref) / 1e6
    if a.time:
        for _ in range(a.warmup):
            run(m, ref, default, x)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(a.runs):
            torch.cuda.synchronize(); t0 = time.perf_counter(); run(m, ref, default, x); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        res.update(time_s=sum(ts) / len(ts), peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9,
                   peak_mem_GiB=torch.cuda.max_memory_allocated() / 2**30,
                   gpu=torch.cuda.get_device_name())
    if a.flops:
        with FlopCounterMode(display=False) as fc:
            run(m, ref, default, x)
        res["tflops"] = fc.get_total_flops() / 1e12
    print(json.dumps(res))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
