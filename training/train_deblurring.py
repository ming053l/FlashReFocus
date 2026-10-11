#!/usr/bin/env python
"""Stage I training (Sec. 3.1): one-step diffusion on Stable Diffusion 2.1 with a pixel LoRA and a detail LoRA, trained
for defocus deblurring. Model and CSD loss come from third_party/PiSA-SR, used unchanged; the LoRAs start from its
released one-step weights (--init_checkpoint). train_deblurring.sh gives the configuration of the released model.

  * data: the mixture of training/configs/deblurring_data.yaml (DPDD, RealBokeh, LFDOF, synthetic blur on DIV2K);
  * pixel phase (steps < --pixel_steps): L2 on the pixel LoRA, optionally with a LoRA on the VAE encoder
    (--vae_lora_rank) and a trainable U-Net conv_in (--train_conv_in);
  * detail phase: L2 + 2 LPIPS + 1 CSD (guidance 7.5, t in [20, 500]) on the detail LoRA, with everything trained in the
    pixel phase frozen. 50% empty prompts. CSD teacher prompts: RAM++ tags of the sharp target; its negative prompt
    drops "over-smooth" (it penalises naturally smooth regions such as sky) and adds defocus terms.

LoRA scaling: peft scales a LoRA update by lora_alpha / r, so a higher-rank LoRA learns more slowly than the rank-4
initialisation; --pixel_lora_scaling and --pixel_lora_a_std keep it at that scale (--detail_lora_* likewise).
Checkpoints store B with PiSA-SR's scaling 8 / r. PiSA-SR calls the detail LoRA "sem"; its adapter names keep it."""
import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from flashrefocus import paths
from deblurring_data import build_deblurring_dataset
from prompt_tagger import RamPlusPromptTagger

paths.use_third_party(paths.PISA_SR)
from pisasr import CSDLoss, PiSASR  # noqa: E402

HERE = Path(__file__).resolve().parent
OFFICIAL_NEG = ("painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, "
                "unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, "
                "jpeg artifacts, deformed, lowres, over-smooth")
DEFOCUS_NEG = OFFICIAL_NEG.replace(", over-smooth", "") + ", bokeh, shallow depth of field, out of focus, defocus blur"


PIXEL_ADAPTERS = ["default_encoder_pix", "default_decoder_pix", "default_others_pix"]
DETAIL_ADAPTERS = ["default_encoder_sem", "default_decoder_sem", "default_others_sem"]
PISA_LORA_ALPHA = 8.0   # peft's default lora_alpha, used by PiSA-SR
VAE_LORA_TARGETS = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=Path, default=paths.WORK / "runs/deblurring")
    p.add_argument("--sd21", default=str(paths.SD21))
    p.add_argument("--init_checkpoint", default=str(paths.PISA_SR / "preset/models/PiSA-SR/pisa_sr.pkl"),
                   help="initial LoRAs ('' = random LoRA init)")
    p.add_argument("--resume", default=None, help="checkpoint model_<step>.pkl to resume from")
    p.add_argument("--data_config", default=str(HERE / "configs/deblurring_data.yaml"))
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--max_train_steps", type=int, default=40000)
    p.add_argument("--pixel_steps", type=int, default=20000, help="steps of the pixel phase; the detail phase follows")
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--lambda_l2", type=float, default=1.0)
    p.add_argument("--lambda_lpips", type=float, default=2.0, help="lambda_LPIPS^det")
    p.add_argument("--lambda_csd", type=float, default=1.0)
    p.add_argument("--csd_guidance", type=float, default=7.5)
    p.add_argument("--csd_min_timestep_ratio", type=float, default=0.02)
    p.add_argument("--csd_max_timestep_ratio", type=float, default=0.5)
    p.add_argument("--csd_negative_prompt", default=DEFOCUS_NEG)
    p.add_argument("--empty_prompt_ratio", type=float, default=0.5)
    p.add_argument("--pixel_lora_rank", type=int, default=4)
    p.add_argument("--detail_lora_rank", type=int, default=4)
    p.add_argument("--pixel_lora_scaling", type=float, default=None, help="default: peft's 8 / r")
    p.add_argument("--pixel_lora_a_std", type=float, default=None, help="std of the LoRA A init (default: peft's 1 / r)")
    p.add_argument("--detail_lora_scaling", type=float, default=None)
    p.add_argument("--detail_lora_a_std", type=float, default=None)
    p.add_argument("--vae_lora_rank", type=int, default=0, help="LoRA rank on the VAE encoder (0 = frozen VAE)")
    p.add_argument("--train_conv_in", action="store_true", help="train the U-Net conv_in weights (pixel phase)")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--checkpoint_every", type=int, default=2500)
    p.add_argument("--val_every", type=int, default=1000)
    return p.parse_args()


def vae_encoder_lora_modules(vae):
    """Every convolution and attention projection of the VAE encoder."""
    return sorted({n[:-len(".weight")] for n, _ in vae.named_parameters()
                   if n.startswith("encoder.") and n.endswith(".weight") and "norm" not in n
                   and any(t in n for t in VAE_LORA_TARGETS)})


def lora_scaling(unet, parameter_name):
    """(scaling, rank) of the LoRA layer that owns parameter `parameter_name` (...lora_B.<adapter>.weight)."""
    module_path, rest = parameter_name.split(".lora_", 1)
    adapter = rest.split(".")[1]
    module = unet.get_submodule(module_path)
    return module.scaling[adapter], module.r[adapter]


def load_unet_lora(unet, state_dict):
    """Copy LoRA weights stored with PiSA-SR's scaling (update = 8 / r_saved * B A). A smaller saved rank is embedded
    exactly into the first ranks of a larger LoRA; B is converted to this model's scaling."""
    copied = embedded = 0
    for n, p in unet.named_parameters():
        if "lora" not in n or n not in state_dict:
            continue
        w = state_dict[n].to(p.dtype)
        if "lora_A" in n:
            p.data[: w.shape[0]].copy_(w)
        else:  # lora_B: [out, r, ...]
            scaling, _ = lora_scaling(unet, n)
            p.data.zero_()
            p.data[:, : w.shape[1]].copy_(w * (PISA_LORA_ALPHA / w.shape[1]) / scaling)
        copied += w.shape == p.shape
        embedded += w.shape != p.shape
    return copied, embedded


def configure_lora(unet, args):
    """Optional scaling and A-init overrides of the pixel and detail LoRAs (see the module docstring)."""
    for _, module in unet.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        for adapter in list(module.lora_A.keys()):
            kind = "pixel" if "pix" in adapter else "detail"
            a_std, scaling = getattr(args, f"{kind}_lora_a_std"), getattr(args, f"{kind}_lora_scaling")
            if a_std:
                torch.nn.init.normal_(module.lora_A[adapter].weight, std=a_std)
            if scaling:
                module.scaling[adapter] = scaling


def conv_in_parameters(unet):
    return [p for n, p in unet.named_parameters() if n.startswith("conv_in.") and "lora" not in n]


def vae_lora_parameters(vae):
    return [p for n, p in vae.named_parameters() if "lora" in n]


def set_pixel_phase_extras_trainable(model, trainable, args):
    for p in vae_lora_parameters(model.vae_fix):
        p.requires_grad = trainable
    for p in conv_in_parameters(model.unet):
        p.requires_grad = trainable and args.train_conv_in


def save_checkpoint(model, path, args, vae_lora_modules):
    """PiSA-SR's checkpoint format (LoRA B converted to its scaling 8 / r) plus the VAE-encoder LoRA and conv_in."""
    model.save_model(path)
    sd = torch.load(path, map_location="cpu")
    for n in list(sd["state_dict_unet"]):
        if "lora_B" in n:
            scaling, r = lora_scaling(model.unet, n)
            sd["state_dict_unet"][n] = sd["state_dict_unet"][n] * (scaling / (PISA_LORA_ALPHA / r))
    if args.vae_lora_rank:
        sd["vae_lora_rank"] = args.vae_lora_rank
        sd["vae_lora_encoder_modules"] = vae_lora_modules
        sd["state_dict_vae"] = {k: v.cpu() for k, v in model.vae_fix.state_dict().items() if "lora" in k}
    if args.train_conv_in:
        sd["conv_in"] = {k: v.detach().cpu() for k, v in model.unet.state_dict().items()
                         if k.startswith("conv_in.") and "lora" not in k}
    torch.save(sd, path)


def validation_pairs():
    """Fixed validation sets: 6 DPDD validation crops (768 x 768, centre) and 6 held-out LFDOF test scenes (the middle
    focal slice, full 1008 x 688)."""
    root = paths.DATA / "dd_dp_dataset_png/val_c"
    dpdd = list(zip(sorted((root / "source").glob("*.png"))[:6], sorted((root / "target").glob("*.png"))[:6]))
    root = paths.DATA / "LFDOF/test_data"
    scenes = sorted(p for p in (root / "input").iterdir() if p.is_dir())[::8][:6]
    lfdof = [(sorted(s.glob("*.png"))[len(list(s.glob("*.png"))) // 2], root / "ground_truth" / f"{s.name}.png")
             for s in scenes]
    return {"val": dpdd, "val_lfdof": lfdof}


def laplacian_variance(x):
    """Laplacian variance of the grey version of a [-1, 1] image tensor [1, 3, H, W]."""
    g = (x[0] * 0.5 + 0.5).clamp(0, 1)
    g = (0.299 * g[0] + 0.587 * g[1] + 0.114 * g[2])[None, None] * 255
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device)[None, None]
    return F.conv2d(g.float(), k).var().item()


def collate_pairs(samples):
    """The four sources return different extra keys; only the blurred / sharp pair is used."""
    return {"source": torch.stack([s["source"] for s in samples]), "target": torch.stack([s["target"] for s in samples])}


def center_crop(img, size=768):
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def enter_phase(model, detail_phase, args):
    """Pixel phase: the pixel LoRA (and the extras) train. Detail phase: the detail LoRA trains on top of it."""
    model.unet.set_adapter(PIXEL_ADAPTERS + DETAIL_ADAPTERS if detail_phase else PIXEL_ADAPTERS)
    model.set_train_sem() if detail_phase else model.set_train_pix()
    set_pixel_phase_extras_trainable(model, not detail_phase, args)
    return (args.lambda_l2, args.lambda_lpips, args.lambda_csd) if detail_phase else (args.lambda_l2, 0.0, 0.0)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    acc = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="fp16",
                      project_config=ProjectConfiguration(project_dir=str(args.output_dir)),
                      kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    set_seed(args.seed + acc.process_index)
    main_process = acc.is_main_process
    if main_process:
        (args.output_dir / "checkpoints").mkdir(exist_ok=True)
        (args.output_dir / "val").mkdir(exist_ok=True)
        json.dump({k: str(v) for k, v in vars(args).items()}, open(args.output_dir / "args.json", "w"), indent=1)
        tb = SummaryWriter(str(args.output_dir / "tb"))

    # model (PiSA-SR's training model), optionally starting from released LoRAs
    pisa_args = SimpleNamespace(pretrained_model_path=args.sd21, pretrained_model_path_csd=args.sd21,
                                lora_rank_unet_pix=args.pixel_lora_rank, lora_rank_unet_sem=args.detail_lora_rank,
                                resume_ckpt=None, timesteps1=1, enable_xformers_memory_efficient_attention=False,
                                min_dm_step_ratio=args.csd_min_timestep_ratio, max_dm_step_ratio=args.csd_max_timestep_ratio,
                                cfg_csd=args.csd_guidance, null_text_ratio=args.empty_prompt_ratio)
    net = PiSASR(pisa_args)
    vae_lora_modules = None
    if args.vae_lora_rank:
        from peft import LoraConfig
        vae_lora_modules = vae_encoder_lora_modules(net.vae_fix)
        net.vae_fix.add_adapter(LoraConfig(r=args.vae_lora_rank, init_lora_weights="gaussian",
                                           target_modules=vae_lora_modules), adapter_name="vae_enc")
    configure_lora(net.unet, args)
    start_step = 0
    init = args.resume or args.init_checkpoint
    if init:
        sd = torch.load(init, map_location="cpu")
        copied, embedded = load_unet_lora(net.unet, sd["state_dict_unet"])
        if "state_dict_vae" in sd and args.vae_lora_rank:
            for n, p in net.vae_fix.named_parameters():
                if n in sd["state_dict_vae"]:
                    p.data.copy_(sd["state_dict_vae"][n])
        if "conv_in" in sd:
            net.unet.load_state_dict(sd["conv_in"], strict=False)
        if main_process:
            print(f"loaded LoRAs from {init} (copied {copied}, rank-embedded {embedded}; "
                  f"VAE LoRA {'state_dict_vae' in sd}, conv_in {'conv_in' in sd})")
        if args.resume:
            start_step = int(Path(args.resume).stem.split("_")[-1])
    csd = CSDLoss(args=pisa_args, accelerator=acc)
    csd.requires_grad_(False)
    lpips_vgg = lpips.LPIPS(net="vgg").to(acc.device).eval()
    lpips_vgg.requires_grad_(False)
    tagger = RamPlusPromptTagger().to(acc.device).eval()
    tagger.requires_grad_(False)

    detail_phase = start_step >= args.pixel_steps
    net.unet.set_adapter(PIXEL_ADAPTERS + DETAIL_ADAPTERS if detail_phase else PIXEL_ADAPTERS)
    params = [p for n, p in net.unet.named_parameters() if "lora" in n] + vae_lora_parameters(net.vae_fix)
    if args.train_conv_in:
        params += conv_in_parameters(net.unet)
    for p in params:   # every parameter ever trained must require grad when DDP builds its reducer
        p.requires_grad = True
    opt = torch.optim.AdamW(params, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    sched = get_scheduler("constant_with_warmup", optimizer=opt, num_warmup_steps=args.lr_warmup_steps * acc.num_processes)

    dataset = build_deblurring_dataset(args.data_config, args.resolution)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                         num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                                         collate_fn=collate_pairs)
    net, opt, loader, sched = acc.prepare(net, opt, loader, sched)
    # optimizer, schedule and grad-scaler state are saved next to each checkpoint (opt_<step>.pt), so a resumed run
    # keeps the Adam moments and does not warm up again
    opt_state = Path(args.resume).with_name(f"opt_{start_step}.pt") if args.resume else None
    if opt_state is not None and opt_state.exists():
        st = torch.load(opt_state, map_location="cpu")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        if acc.scaler is not None and st.get("scaler") is not None:
            acc.scaler.load_state_dict(st["scaler"])
    elif main_process and args.resume:
        print(f"no optimizer state next to {args.resume}; Adam and the lr warm-up start fresh")
    lambdas = enter_phase(acc.unwrap_model(net), detail_phase, args)
    if main_process:
        print(f"dataset {len(dataset)} samples; effective batch "
              f"{args.train_batch_size * args.gradient_accumulation_steps * acc.num_processes}; start step {start_step}")

    vals = validation_pairs()
    step, t_last, done = start_step, time.time(), False
    while not done:
        for batch in loader:
            with acc.accumulate(net):
                x_src, x_tgt = batch["source"] * 2 - 1, batch["target"] * 2 - 1
                # the released training rescaled the [0, 1] targets as if they were in [-1, 1] before tagging
                prompts = tagger.prompts((batch["target"].float() + 1.0) / 2.0)
                n = x_src.shape[0]
                prompt_batch = {"prompt": prompts, "neg_prompt": [args.csd_negative_prompt] * n, "null_prompt": [""] * n}
                pred, latent, prompt_embeds, neg_prompt_embeds = net(x_src, x_tgt, batch=prompt_batch, args=pisa_args)
                zero = torch.zeros((), device=pred.device)
                l_l2 = F.mse_loss(pred.float(), x_tgt.float()) * lambdas[0]
                l_lpips = lpips_vgg(pred.float(), x_tgt.float()).mean() * lambdas[1] if lambdas[1] > 0 else zero
                l_csd = csd.cal_csd(latent, prompt_embeds, neg_prompt_embeds, pisa_args) * lambdas[2] if lambdas[2] > 0 else zero
                loss = l_l2 + l_lpips + l_csd
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(params, args.max_grad_norm)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if not acc.sync_gradients:
                continue
            step += 1
            if step >= args.pixel_steps and not detail_phase:
                detail_phase = True
                lambdas = enter_phase(acc.unwrap_model(net), detail_phase, args)
                if main_process:
                    print(f"step {step}: detail phase", flush=True)
            if main_process:
                for name, value in (("total", loss), ("l2", l_l2), ("lpips", l_lpips), ("csd", l_csd)):
                    tb.add_scalar(f"loss/{name}", value.item(), step)
                tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
                if step % 50 == 0:
                    dt = (time.time() - t_last) / 50; t_last = time.time()
                    print(f"step {step} loss {loss.item():.4f} l2 {l_l2.item():.4f} lpips {l_lpips.item():.4f} "
                          f"csd {l_csd.item():.4f} {dt:.2f}s/step eta {dt * (args.max_train_steps - step) / 3600:.1f}h",
                          flush=True)
                if step % args.checkpoint_every == 0 or step in (args.pixel_steps, args.max_train_steps):
                    ckpt_dir = args.output_dir / "checkpoints"
                    save_checkpoint(acc.unwrap_model(net), str(ckpt_dir / f"model_{step}.pkl"), args, vae_lora_modules)
                    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(),
                                "scaler": acc.scaler.state_dict() if acc.scaler is not None else None, "step": step},
                               ckpt_dir / f"opt_{step}.pt")
                if step % args.val_every == 0 or step == 1:
                    validate(acc.unwrap_model(net), vals, lpips_vgg, tb, step, args)
            acc.wait_for_everyone()
            if step >= args.max_train_steps:
                done = True
                break
    if main_process:
        tb.close()


@torch.no_grad()
def validate(model, vals, lpips_vgg, tb, step, args):
    """PSNR, LPIPS and sharpness (Laplacian variance of the output / of the ground truth; < 1 means blurrier) of the
    one-pass model on the validation sets."""
    model.unet.eval()
    for tag, pairs in vals.items():
        psnrs, lps, sharpness, grid = [], [], [], []
        for i, (s, t) in enumerate(pairs):
            src, tgt = Image.open(s).convert("RGB"), Image.open(t).convert("RGB")
            if tag == "val":
                src, tgt = center_crop(src), center_crop(tgt)
            else:
                w, h = src.size
                src, tgt = src.crop((0, 0, w - w % 8, h - h % 8)), tgt.crop((0, 0, w - w % 8, h - h % 8))
            x = TF.to_tensor(src)[None].cuda() * 2 - 1
            y = TF.to_tensor(tgt)[None].cuda() * 2 - 1
            z = model.vae_fix.encode(x).latent_dist.mode() * model.vae_fix.config.scaling_factor
            emb = model.encode_prompt([""])
            out = model.vae_fix.decode((z - model.unet(z, model.timesteps1, encoder_hidden_states=emb).sample)
                                       / model.vae_fix.config.scaling_factor).sample.clamp(-1, 1)
            psnrs.append(10 * math.log10(1 / max(F.mse_loss(out * 0.5 + 0.5, y * 0.5 + 0.5).item(), 1e-10)))
            lps.append(lpips_vgg(out.float(), y.float()).mean().item())
            sharpness.append(laplacian_variance(out) / max(laplacian_variance(y), 1e-6))
            if i < 3 and tag == "val":
                grid.append(torch.cat([x, out, y], dim=3)[0] * 0.5 + 0.5)
        tb.add_scalar(f"{tag}/psnr", float(np.mean(psnrs)), step)
        tb.add_scalar(f"{tag}/lpips", float(np.mean(lps)), step)
        tb.add_scalar(f"{tag}/sharpness", float(np.mean(sharpness)), step)
        if grid:
            tb.add_image(f"{tag}/input_output_gt", torch.cat(grid, dim=1).clamp(0, 1), step)
            TF.to_pil_image(torch.cat(grid, dim=1).clamp(0, 1).cpu()).resize((1152, 1152)).save(
                args.output_dir / "val" / f"{step:06d}.jpg")
    model.unet.train()


if __name__ == "__main__":
    main()
