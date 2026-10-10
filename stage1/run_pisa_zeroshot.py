#!/usr/bin/env python
"""Stage I inference without the refiner on DPDD / RealDOF; with the default --ckpt, the released PiSA-SR as a
zero-shot deblurring baseline.

PiSADeterministic wraps PiSA-SR's evaluation model with three changes for deblurring:
  * VAE encoding uses latent_dist.mode() instead of .sample(), so the output is deterministic (the restored
    all-in-focus image is reused by every focal edit);
  * no latent or VAE tiling: the whole image is processed at once;
  * no upscaling; inputs are only cropped to a multiple of 8.
Pairs come from pairs/<set>_test.json (paths relative to the data root); outputs are written as {index:04d}.png."""
import argparse
import json
import time
from pathlib import Path
import sys

import torch
import torchvision.transforms.functional as TF
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repository root
import paths  # noqa: E402
paths.add_to_path()  # third_party/PiSA-SR, and RAM++ ahead of PiSA-SR's own `ram`
from pisasr import PiSASR_eval  # noqa: E402
from src.my_utils.wavelet_color_fix import adain_color_fix  # noqa: E402

PAIRS = {"dpdd": HERE / "pairs/dpdd_test.json", "realdof": HERE / "pairs/realdof_test.json"}   # paths relative to paths.DATA


class PiSADeterministic(PiSASR_eval):
    def _load_pretrained_weights(self, pretrained_path):
        """PiSA-SR loader, plus the optional Stage I extras (trained conv_in, VAE-encoder LoRA merged into the VAE)."""
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
    ap.add_argument("--scale", type=float, default=1.0, help="deblur at this input scale, then resize back")
    ap.add_argument("--ram", action="store_true", help="prompt = RAM++ tags of the blurred input")
    ap.add_argument("--ckpt", default=str(paths.PISA_SR / "preset/models/PiSA-SR/pisa_sr.pkl"),
                    help="LoRA checkpoint (default: released PiSA-SR; or a fine-tuned model_*.pkl)")
    a = ap.parse_args()
    args = argparse.Namespace(
        pretrained_model_path=str(paths.SD21),
        pretrained_path=a.ckpt,
        lambda_pix=a.lambda_pix, lambda_sem=a.lambda_sem, mixed_precision="fp16", default=a.default,
        vae_decoder_tiled_size=100000, vae_encoder_tiled_size=100000,   # effectively no VAE tiling
        latent_tiled_size=a.latent_tile, latent_tiled_overlap=a.latent_overlap, seed=42)  # default: no latent tiling
    model = PiSADeterministic(args)
    model.set_eval()
    tagger = None
    if a.ram:
        from ram_plus_tagger import RamPlusTagger
        tagger = RamPlusTagger(checkpoint_path=str(paths.RAM_PLUS),
                               image_size=384, vit="swin_l", fallback_prompt="a sharp, clear image", prompt_threshold=0.7,
                               max_tags=50).cuda().eval()
    samples = json.load(open(PAIRS[a.dataset]))[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    times = []
    for s in samples:
        img = Image.open(paths.DATA / s["source"]).convert("RGB")
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
