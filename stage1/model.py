"""Stage I model: loading, the one-step restoration and image I/O, shared by every Stage I script and the pipeline.

    model = load_stage1(omega=1.25)               # one-step diffusion with the pixel and detail LoRAs
    deblur = Deblurrer(model, load_refiner())     # decoder refiner (None: SD2.1 VAE decoder)
    y = deblur(x)                                 # x, y: 1x3xHxW in [-1, 1], H and W multiples of 8

omega (the detail weight) can be changed at any time with `deblur.omega = ...` as long as the model was loaded with
omega != 1 (two U-Net passes); omega = 1 is a single pass with the merged LoRAs.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import paths

paths.add_to_path()
from pisasr import PiSASR_eval  # noqa: E402

NO_TILING = 100000   # tile sizes above any input: the whole image is processed at once


class Stage1Model(PiSASR_eval):
    """PiSA-SR's evaluation model (third_party/PiSA-SR) with the changes needed for deblurring:
      * deterministic VAE encoding (latent_dist.mode()), so the restored image is reproducible;
      * the Stage I extras of our checkpoints (trained U-Net conv_in, LoRA on the VAE encoder);
      * the merged pixel + detail weights of the two-pass (omega != 1) mode are built once and reused, instead of being
        re-added for every image (same values, no per-call allocation of the merged weights)."""

    def _load_pretrained_weights(self, pretrained_path):
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

    def _apply_lora_delta(self):
        if getattr(self, "_merged_weight", None) is None:
            self._merged_weight = {n: self.lora_deltas_sem[n] + w if n in self.lora_deltas_sem else w
                                   for n, w in self.ori_unet_weight.items()}
        for name, param in self.unet.named_parameters():
            param.data = self._merged_weight[name]

    def predict(self, z, prompt_embeds, single_pass):
        """U-Net prediction on the whole latent (no tiling, no logging)."""
        return self._predict_no_tiling(z, prompt_embeds, single_pass)

    @torch.no_grad()
    def forward(self, default, c_t, prompt=None):
        c_t = c_t.to(dtype=self.weight_dtype)
        prompt_embeds = self.encode_prompt([prompt]).to(dtype=self.weight_dtype)
        z = self.vae.encode(c_t).latent_dist.mode() * self.vae.config.scaling_factor
        model_pred = self._process_latents(z, prompt_embeds, default)
        return self.vae.decode((z - model_pred) / self.vae.config.scaling_factor).sample.clamp(-1, 1)


def load_stage1(ckpt=paths.STAGE1, sd21=paths.SD21, omega=1.0, lambda_pix=1.0, precision="fp16",
                latent_tile=NO_TILING, latent_overlap=0, merge_vae_lora=False):
    """Stage I in eval mode. omega != 1 (or lambda_pix != 1) selects the two-pass mode, in which omega can be changed
    later; omega = 1 merges the detail LoRA into a single pass.
    merge_vae_lora folds the VAE-encoder LoRA into the encoder weights: the encoder runs ~20% faster, but the output is
    no longer bit-identical to the reported results (about 60 dB PSNR apart), so it is off by default."""
    single = omega == 1.0 and lambda_pix == 1.0
    args = argparse.Namespace(pretrained_model_path=str(sd21), pretrained_path=str(ckpt), lambda_pix=lambda_pix,
                              lambda_sem=omega, mixed_precision=precision, default=single,
                              vae_decoder_tiled_size=NO_TILING, vae_encoder_tiled_size=NO_TILING,
                              latent_tiled_size=latent_tile, latent_tiled_overlap=latent_overlap, seed=42)
    model = Stage1Model(args)
    model.set_eval()
    model.single_pass = single
    if merge_vae_lora:
        from peft.tuners.lora import LoraLayer
        for mod in model.vae.modules():
            if isinstance(mod, LoraLayer):
                mod.merge()
    return model


def load_refiner(path=paths.REFINER, sd15=paths.SD15, device="cuda", dtype=torch.bfloat16, check=True):
    """Decoder refiner (SD1.5 VAE decoder with fusion modules), in bf16 as trained. Its VAE encoder is not part of the
    checkpoint (only the first two down blocks are used, from the SD1.5 VAE)."""
    from diffusers import AutoencoderKL
    from refiner.vae_refiner_wo_d import VAEMultiScaleFusionDecoder
    ref = VAEMultiScaleFusionDecoder(AutoencoderKL.from_pretrained(str(sd15), subfolder="vae"), depth=2,
                                     map_hidden_channels=32, use_pixel_weighting=False, use_high_freq_info=True)
    res = ref.load_state_dict(torch.load(str(path), map_location="cpu"), strict=False)
    if check:
        missing = [k for k in res.missing_keys if not k.startswith(("vae.encoder", "vae.quant_conv", "vae.post_quant_conv"))]
        assert not missing and not res.unexpected_keys, (missing[:5], res.unexpected_keys[:5])
    return ref.to(device, dtype=dtype).eval()


def stage1_params(model, refiner=None):
    """Parameters of everything a configuration loads: text encoder, U-Net with merged LoRAs, VAE, and the refiner
    without the VAE encoder it does not run."""
    n = sum(p.numel() for m in (model.text_encoder, model.unet, model.vae) for p in m.parameters())
    if refiner is not None:
        n += sum(v.numel() for k, v in refiner.state_dict().items() if not k.startswith(("vae.encoder", "vae.quant_conv")))
    return n


class Deblurrer:
    """Blurred image -> restored image with a loaded Stage I model and an optional refiner. The empty-prompt
    embedding is computed once."""

    def __init__(self, model, refiner=None):
        self.m, self.ref = model, refiner
        self.wd = model.weight_dtype
        self.sf = model.vae.config.scaling_factor
        with torch.no_grad():
            self.emb = model.encode_prompt([""]).to(dtype=self.wd)

    @property
    def omega(self):
        return float(self.m.lambda_sem)

    @omega.setter
    def omega(self, value):
        assert not self.m.single_pass or value == 1.0, "load the model with omega != 1 to change omega"
        self.m.lambda_sem = torch.tensor([float(value)], device=self.m.device)

    @property
    def single_pass(self):
        """One U-Net pass when the model was loaded with omega = 1, or when omega is set back to 1."""
        return self.m.single_pass or (float(self.m.lambda_sem) == 1.0 and float(self.m.lambda_pix) == 1.0)

    @torch.no_grad()
    def latent(self, x):
        """Restored latent (unscaled, as the VAE decoder takes it)."""
        z = self.m.vae.encode(x.to(self.wd)).latent_dist.mode() * self.sf
        emb = self.emb.expand(z.shape[0], -1, -1)
        return (z - self.m.predict(z, emb, self.single_pass)) / self.sf

    @torch.no_grad()
    def __call__(self, x, refiner_tile=0, refiner_overlap=64):
        """Restored image in [-1, 1] (not clamped); refiner_tile > 0 runs the refiner on overlapping tiles."""
        lat = self.latent(x)
        if self.ref is None:
            return self.m.vae.decode(lat).sample
        rd = next(self.ref.parameters()).dtype
        x, lat = x.to(rd), lat.to(rd)
        if refiner_tile > 0:
            return tiled_refine(self.ref, x, lat, refiner_tile, refiner_overlap)
        return self.ref(x, latent=lat, sample_posterior=False)


@torch.no_grad()
def tiled_refine(refiner, x, lat, tile, overlap):
    """Refiner on overlapping pixel tiles (multiples of 8), blended with linear ramps."""
    _, _, H, W = x.shape
    tile, overlap = tile - tile % 8, overlap - overlap % 8
    starts = lambda n: sorted({min(p, max(0, n - tile)) for p in range(0, max(1, n - overlap), tile - overlap)})
    out = torch.zeros_like(x, dtype=torch.float32); wsum = torch.zeros((1, 1, H, W), device=x.device)
    ramp = torch.linspace(0, 1, overlap + 2, device=x.device)[1:-1]
    for t in starts(H):
        for l in starts(W):
            b, r = min(t + tile, H), min(l + tile, W)
            rec = refiner(x[:, :, t:b, l:r], latent=lat[:, :, t // 8:b // 8, l // 8:r // 8], sample_posterior=False).float()
            wy = torch.ones(b - t, device=x.device); wx = torch.ones(r - l, device=x.device)
            if overlap:
                if t > 0: wy[:overlap] = ramp
                if b < H: wy[-overlap:] = ramp.flip(0)
                if l > 0: wx[:overlap] = ramp
                if r < W: wx[-overlap:] = ramp.flip(0)
            wgt = (wy[:, None] * wx[None, :])[None, None]
            out[:, :, t:b, l:r] += rec * wgt; wsum[:, :, t:b, l:r] += wgt
    return out / wsum.clamp_min(1e-6)


def read_image(path, device="cuda", multiple=8):
    """RGB image -> 1x3xHxW in [-1, 1] on `device`, cropped (bottom/right) to a multiple of `multiple`.
    Converted exactly as torchvision's to_tensor (contiguous CHW, float division by 255 on the CPU), so every script
    feeds Stage I the same values."""
    return to_unit(Image.open(path).convert("RGB"), multiple).to(device) * 2 - 1


def to_unit(img, multiple=1):
    """PIL RGB image -> 1x3xHxW float in [0, 1] on the CPU, as torchvision's to_tensor, cropped to a multiple."""
    arr = np.asarray(img)
    h, w = arr.shape[0] - arr.shape[0] % multiple, arr.shape[1] - arr.shape[1] % multiple
    return torch.from_numpy(np.ascontiguousarray(arr[:h, :w])).permute(2, 0, 1).contiguous().float().div(255)[None]


def to_uint8(y):
    """[-1, 1] image tensor 1x3xHxW -> HxWx3 uint8, quantised by truncation as torchvision's to_pil_image."""
    return ((y[0].float() * 0.5 + 0.5).clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def save_image(y, path, compress_level=1):
    Image.fromarray(to_uint8(y)).save(path, compress_level=compress_level)
