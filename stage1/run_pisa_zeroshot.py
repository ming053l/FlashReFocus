#!/usr/bin/env python
"""Zero-shot PiSA-SR as a Stage-I deblurrer on DPDD / RealDOF (Phase 1).

The official PiSA-SR code is used unchanged (imported from ./PiSA-SR); three defaults that do not
fit our task are overridden here:
  * VAE encoding uses latent_dist.mode() instead of .sample(), so the output is deterministic
    (the restored AiF image is reused across focal edits);
  * no latent or VAE tiling: the whole image is processed at once, as for PASD in Table 1;
  * upscale = 1 (deblurring, not super-resolution); inputs are only cropped to a multiple of 8.
Input/GT pairs and output names (0000.png, ...) are taken from the PASD run of Table 1, so both
methods are scored by MyAutoRefocus/evaluate_folders.py on identical pairs.
"""
import argparse
import json
import time
from pathlib import Path
import sys

import torch
import torchvision.transforms.functional as TF
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "PiSA-SR"))
sys.path.insert(0, "/ssd1/chingheng/Defocus/MyAutoRefocus")  # its `ram` package (RAM++) must win over PiSA-SR's RAM
from pisasr import PiSASR_eval  # noqa: E402
from src.my_utils.wavelet_color_fix import adain_color_fix  # noqa: E402

PASD_RUNS = Path("/ssd1/chingheng/Defocus/MyAutoRefocus/runs")
PAIRS = {
    "dpdd": PASD_RUNS / "dpdd_ans_sweep/n400/two_stage_dpdd_cfg1.0_ans400_tile320/validation/step_000000/metrics.json",
    "realdof": PASD_RUNS / "realdof_ans_sweep/n400/two_stage_realdof_cfg1.0_ans400_tile320/validation/step_000000/metrics.json",
}


class PiSADeterministic(PiSASR_eval):
    def _load_pretrained_weights(self, pretrained_path):
        """Official loader, plus the optional v2 extras (trained conv_in, VAE-encoder LoRA merged into the VAE)."""
        sd = torch.load(pretrained_path, map_location="cpu")
        if "conv_in" in sd:  # base weights, before the pixel LoRA is merged on top
            self.unet.load_state_dict(sd["conv_in"], strict=False)
        self._load_and_save_ckpt_from_state_dict(sd)
        if "state_dict_vae" in sd:
            from peft import LoraConfig
            self.vae.add_adapter(LoraConfig(r=sd["vae_lora_rank"], init_lora_weights="gaussian",
                                            target_modules=sd["vae_lora_encoder_modules"]), adapter_name="vae_enc")
            for n, p in self.vae.named_parameters():
                if n in sd["state_dict_vae"]:
                    p.data.copy_(sd["state_dict_vae"][n])

    @torch.no_grad()
    def forward(self, default, c_t, prompt=None):
        c_t = c_t.to(dtype=self.weight_dtype)
        prompt_embeds = self.encode_prompt([prompt]).to(dtype=self.weight_dtype)
        z = self.vae.encode(c_t).latent_dist.mode() * self.vae.config.scaling_factor
        model_pred = self._process_latents(z, prompt_embeds, default)
        return self.vae.decode((z - model_pred) / self.vae.config.scaling_factor).sample.clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(PAIRS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--align", choices=["nofix", "adain"], default="nofix")
    ap.add_argument("--default", action="store_true", help="single UNet pass (lambda_pix = lambda_sem = 1)")
    ap.add_argument("--lambda_pix", type=float, default=1.0)
    ap.add_argument("--lambda_sem", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--latent_tile", type=int, default=100000, help="latent tile size (official PiSA-SR: 96); default = whole image")
    ap.add_argument("--latent_overlap", type=int, default=0, help="latent tile overlap (official: 32)")
    ap.add_argument("--scale", type=float, default=1.0, help="deblur at this input scale, then resize back (diagnostic)")
    ap.add_argument("--ram", action="store_true", help="prompt = RAM++ tags of the blurred input (as in PASD Stage I)")
    ap.add_argument("--ckpt", default=str(HERE / "PiSA-SR/preset/models/PiSA-SR/pisa_sr.pkl"),
                    help="LoRA checkpoint (default: released PiSA-SR; or a fine-tuned model_*.pkl)")
    a = ap.parse_args()
    args = argparse.Namespace(
        pretrained_model_path="/ssd1/chingheng/Defocus/stable-diffusion-2-1-base",
        pretrained_path=a.ckpt,
        lambda_pix=a.lambda_pix, lambda_sem=a.lambda_sem, mixed_precision="fp16", default=a.default,
        vae_decoder_tiled_size=100000, vae_encoder_tiled_size=100000,   # effectively no VAE tiling
        latent_tiled_size=a.latent_tile, latent_tiled_overlap=a.latent_overlap, seed=42)  # default: no latent tiling
    model = PiSADeterministic(args)
    model.set_eval()
    tagger = None
    if a.ram:
        from ram_plus_tagger import RamPlusTagger
        tagger = RamPlusTagger(checkpoint_path="/ssd1/chingheng/Defocus/SOTA/recognize-anything/pretrained/ram_plus_swin_large_14m.pth",
                               image_size=384, vit="swin_l", fallback_prompt="a sharp, clear image", prompt_threshold=0.7,
                               max_tags=50).cuda().eval()
    samples = json.load(open(PAIRS[a.dataset]))["samples"][: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    times = []
    for s in samples:
        img = Image.open(s["source_path"]).convert("RGB")
        w, h = img.size
        img = img.crop((0, 0, w - w % 8, h - h % 8))
        full = img.size
        if a.scale != 1.0:
            img = img.resize((round(full[0] * a.scale / 8) * 8, round(full[1] * a.scale / 8) * 8), Image.BICUBIC)
        c_t = TF.to_tensor(img).unsqueeze(0).cuda() * 2 - 1
        torch.cuda.synchronize(); t0 = time.perf_counter()
        prompt = tagger.predict_batch(c_t.float() * 0.5 + 0.5).prompt_texts[0] if tagger else ""
        out = model(a.default, c_t, prompt=prompt)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        pil = TF.to_pil_image((out[0].float() * 0.5 + 0.5).clamp(0, 1).cpu())
        if pil.size != full:
            pil = pil.resize(full, Image.BICUBIC)
        if a.align == "adain":
            pil = adain_color_fix(target=pil, source=img)
        pil.save(a.out / f"{s['index']:04d}.png")
    print(f"{a.dataset} {len(samples)} images, mean time {sum(times[3:]) / max(1, len(times) - 3):.3f}s (excluding first 3)")


if __name__ == "__main__":
    main()
