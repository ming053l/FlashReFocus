#!/usr/bin/env python
"""Stage I inference with the decoder refiner: PiSA-SR restores the latent (omega = --lambda_sem; 1.25 for the
released model) and the refiner decodes it in place of the SD2.1 decoder, guided by the blurred input.

Pairs come from pairs/<set>_test.json (paths relative to --data_root); outputs are written as {index:04d}.png."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repository root
import paths  # noqa: E402
paths.add_to_path()
from run_pisa_zeroshot import PiSADeterministic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--data_root", default=str(paths.DATA))
    ap.add_argument("--ckpt", required=True, help="PiSA-SR checkpoint (.pkl)")
    ap.add_argument("--refiner", default=None, help="vae_refiner.pt; omit to decode with the SD2.1 VAE (plain PiSA-SR)")
    ap.add_argument("--sd21", default=str(paths.SD21))
    ap.add_argument("--sd15", default=str(paths.SD15))
    ap.add_argument("--lambda_pix", type=float, default=1.0)
    ap.add_argument("--lambda_sem", type=float, default=1.0, help="!= 1 uses PiSA-SR's adjustable mode (two UNet passes)")
    ap.add_argument("--sem_scale", type=float, default=1.0,
                    help="scale the merged semantic-LoRA update in weight space (one UNet pass; alternative to --lambda_sem)")
    ap.add_argument("--refiner_tile", type=int, default=0, help="pixel tile for the refiner (0 = whole image)")
    ap.add_argument("--refiner_overlap", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    default = a.lambda_pix == 1.0 and a.lambda_sem == 1.0
    args = argparse.Namespace(pretrained_model_path=a.sd21, pretrained_path=a.ckpt, lambda_pix=a.lambda_pix,
                              lambda_sem=a.lambda_sem, mixed_precision="fp16", default=default, vae_decoder_tiled_size=100000,
                              vae_encoder_tiled_size=100000, latent_tiled_size=100000, latent_tiled_overlap=0, seed=42)
    model = PiSADeterministic(args)
    model.set_eval()
    if a.sem_scale != 1.0:
        assert default, "--sem_scale replaces --lambda_sem; use one of them"
        # official loader: W = W_pix + delta_sem (merged); ori_unet_weight = W_pix  ->  W_pix + s * delta_sem
        for n, p in model.unet.named_parameters():
            o = model.ori_unet_weight[n]
            p.data = o + a.sem_scale * (p.data - o)
    refiner = None
    if a.refiner:
        from refiner.vae_refiner_wo_d import VAEMultiScaleFusionDecoder
        vae15 = AutoencoderKL.from_pretrained(a.sd15, subfolder="vae")
        refiner = VAEMultiScaleFusionDecoder(vae15, depth=2, map_hidden_channels=32, use_pixel_weighting=False,
                                             use_high_freq_info=True)
        res = refiner.load_state_dict(torch.load(a.refiner, map_location="cpu"), strict=False)
        missing = [k for k in res.missing_keys if not k.startswith("vae.encoder") and not k.startswith("vae.quant_conv")
                   and not k.startswith("vae.post_quant_conv")]
        print(f"refiner loaded: missing (non-encoder) {missing[:5]} unexpected {res.unexpected_keys[:5]}")
        refiner = refiner.to("cuda", dtype=torch.bfloat16).eval()   # the refiner was trained in bf16

    a.out.mkdir(parents=True, exist_ok=True)
    sf = model.vae.config.scaling_factor
    times = []
    for s in json.load(open(a.pairs)):
        img = Image.open(Path(a.data_root) / s["source"]).convert("RGB")
        w, h = img.size
        img = img.crop((0, 0, w - w % 8, h - h % 8))
        x = TF.to_tensor(img).unsqueeze(0).cuda() * 2 - 1
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            if refiner is None:
                out = model(default, x, prompt="")
            else:
                c = x.to(model.weight_dtype)
                emb = model.encode_prompt([""]).to(dtype=model.weight_dtype)
                z = model.vae.encode(c).latent_dist.mode() * sf
                z_out = z - model._process_latents(z, emb, default)
                xb, lat = x.to(torch.bfloat16), (z_out / sf).to(torch.bfloat16)
                if a.refiner_tile <= 0:
                    out = refiner(xb, latent=lat, sample_posterior=False)
                else:
                    out = tiled_refine(refiner, xb, lat, a.refiner_tile, a.refiner_overlap)
                out = out.clamp(-1, 1)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        TF.to_pil_image((out[0].float() * 0.5 + 0.5).clamp(0, 1).cpu()).save(a.out / f"{s['index']:04d}.png")
    print(f"{a.pairs.stem}: {len(times)} images, mean time {sum(times[3:]) / max(1, len(times) - 3):.3f}s (excluding first 3)")


def tiled_refine(refiner, x, lat, tile, overlap):
    """Overlapping pixel tiles (multiples of 8) blended with linear ramps, as validation.run_vae_refiner_inference."""
    _, _, H, W = x.shape
    tile, overlap = tile - tile % 8, overlap - overlap % 8
    starts = lambda n: sorted({min(p, max(0, n - tile)) for p in range(0, max(1, n - overlap), tile - overlap)})
    out = torch.zeros_like(x, dtype=torch.float32); wsum = torch.zeros((1, 1, H, W), device=x.device)
    for t in starts(H):
        for l in starts(W):
            b, r = min(t + tile, H), min(l + tile, W)
            rec = refiner(x[:, :, t:b, l:r], latent=lat[:, :, t // 8:b // 8, l // 8:r // 8], sample_posterior=False).float()
            wy = torch.ones(b - t, device=x.device); wx = torch.ones(r - l, device=x.device)
            if overlap:
                ramp = torch.linspace(0, 1, overlap + 2, device=x.device)[1:-1]
                if t > 0: wy[:overlap] = ramp
                if b < H: wy[-overlap:] = ramp.flip(0)
                if l > 0: wx[:overlap] = ramp
                if r < W: wx[-overlap:] = ramp.flip(0)
            wgt = (wy[:, None] * wx[None, :])[None, None]
            out[:, :, t:b, l:r] += rec * wgt; wsum[:, :, t:b, l:r] += wgt
    return out / wsum.clamp_min(1e-6)


if __name__ == "__main__":
    main()
