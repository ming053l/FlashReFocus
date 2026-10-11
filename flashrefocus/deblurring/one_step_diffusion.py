"""Stage I one-step diffusion model (Sec. 3.1): Stable Diffusion 2.1 with a pixel LoRA and a detail LoRA, adapted from
PiSA-SR (third_party/PiSA-SR). The restored latent is z_hat = z_blur - eps(z_blur), with eps the U-Net prediction at
timestep 1 and an empty prompt, and

    eps = eps_pix + omega * (eps_pix+det - eps_pix)

where eps_pix uses the pixel LoRA alone, eps_pix+det both LoRAs and omega is the detail weight. omega = 1 is a single
U-Net pass with both LoRAs merged; any other omega needs both predictions (two passes).

PiSA-SR names the detail LoRA "sem" (semantic) and omega "lambda_sem"; these names remain in its checkpoint keys."""
import argparse

import torch

from .. import paths

NO_TILING = 100000   # tile sizes above any input: the whole image is processed at once


def _pisa_sr_eval_class():
    paths.use_third_party(paths.PISA_SR)
    from pisasr import PiSASR_eval
    return PiSASR_eval


class OneStepDiffusion(_pisa_sr_eval_class()):
    """PiSA-SR's evaluation model with the changes needed for deblurring:
      * deterministic VAE encoding (latent_dist.mode()), so the restored image is reproducible;
      * the Stage I extras of our checkpoints (trained U-Net conv_in, LoRA on the VAE encoder);
      * the merged pixel + detail weights of the two-pass mode are built once and reused."""

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
        # back to the pixel + detail weights after the pixel-only pass, without re-adding the deltas every call
        if getattr(self, "_merged_weight", None) is None:
            self._merged_weight = {n: self.lora_deltas_sem[n] + w if n in self.lora_deltas_sem else w
                                   for n, w in self.ori_unet_weight.items()}
        for name, param in self.unet.named_parameters():
            param.data = self._merged_weight[name]

    @property
    def detail_weight(self):
        """omega"""
        return float(self.lambda_sem)

    @detail_weight.setter
    def detail_weight(self, omega):
        assert not self.single_pass or omega == 1.0, "load the model with detail_weight != 1 to change it"
        self.lambda_sem = torch.tensor([float(omega)], device=self.device)

    @property
    def uses_single_pass(self):
        """One U-Net pass when loaded with omega = 1, or when omega is set back to 1."""
        return self.single_pass or float(self.lambda_sem) == 1.0

    @torch.no_grad()
    def empty_prompt_embedding(self):
        return self.encode_prompt([""]).to(dtype=self.weight_dtype)

    def predict_noise(self, z, prompt_embeds):
        """eps for the latent z (whole latent, no tiling)."""
        return self._predict_no_tiling(z, prompt_embeds, self.uses_single_pass)


def load_one_step_diffusion(checkpoint=paths.STAGE1, sd21=paths.SD21, detail_weight=1.0, precision="fp16",
                            merge_vae_lora=False):
    """Stage I in eval mode. detail_weight != 1 selects the two-pass mode, in which it can be changed later; 1 merges
    the detail LoRA into a single pass.
    merge_vae_lora folds the VAE-encoder LoRA into the encoder weights: the encoder runs ~20% faster, but the output is
    no longer bit-identical to the reported results (about 60 dB PSNR apart), so it is off by default."""
    single = detail_weight == 1.0
    args = argparse.Namespace(pretrained_model_path=str(sd21), pretrained_path=str(checkpoint), lambda_pix=1.0,
                              lambda_sem=detail_weight, mixed_precision=precision, default=single,
                              vae_decoder_tiled_size=NO_TILING, vae_encoder_tiled_size=NO_TILING,
                              latent_tiled_size=NO_TILING, latent_tiled_overlap=0, seed=42)
    model = OneStepDiffusion(args)
    model.set_eval()
    model.single_pass = single
    if not single:
        # Use the pixel + detail weights that every two-pass call restores after its pixel-only pass from the start.
        # peft's own merge differs from them in the last bits, which made the first image restored differ slightly
        # (up to 2/255) from later ones.
        model._apply_lora_delta()
    if merge_vae_lora:
        from peft.tuners.lora import LoraLayer
        for mod in model.vae.modules():
            if isinstance(mod, LoraLayer):
                mod.merge()
    return model


def stage1_parameter_count(model, refiner=None):
    """Parameters of everything a configuration runs: text encoder, U-Net with merged LoRAs, VAE, and the refiner
    without the VAE encoder it does not run."""
    n = sum(p.numel() for m in (model.text_encoder, model.unet, model.vae) for p in m.parameters())
    if refiner is not None:
        n += sum(v.numel() for k, v in refiner.state_dict().items() if not k.startswith(("vae.encoder", "vae.quant_conv")))
    return n
