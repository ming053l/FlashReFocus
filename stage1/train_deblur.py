#!/usr/bin/env python
"""Stage I training: one-step diffusion on Stable Diffusion 2.1 with a pixel LoRA and a detail LoRA, trained for defocus
deblurring. The model and CSD loss code come from third_party/PiSA-SR and are used unchanged; the LoRAs are initialised
from its pretrained one-step weights (--init_pkl).
  * data: paired defocus mixture of DPDD (0.3), RealBokeh (0.2), LFDOF (0.3) and synthetic spatially varying blur on
    DIV2K (0.2), see common/dpdd_dataset.py and configs/stage1_data.yaml;
  * CSD teacher prompts: RAM++ tags of the sharp target;
  * CSD negative prompt without "over-smooth" (it penalises naturally smooth regions such as sky) and with defocus
    terms, so CSD pushes away from defocus blur rather than from smoothness.

Pixel stage (--pix_steps steps): L2 on the pixel LoRA, optionally with a LoRA on the VAE encoder (--vae_lora_rank)
and a trainable U-Net conv_in (--train_conv_in). Detail stage: L2 + 2 LPIPS + 1 CSD (cfg 7.5, t in [20, 500]) on the
detail LoRA ("sem" in the option names), with everything trained in the pixel stage frozen. 50% null prompt, lr 5e-5.

LoRA scaling: peft scales a LoRA update by lora_alpha / r, so a higher-rank LoRA learns more slowly than the rank-4
initialisation; --pix_lora_scaling and --pix_lora_a_std keep it at that scale (--sem_lora_* likewise). Checkpoints
store B with scaling 8 / r.

Optional, off by default: alternating pixel and detail stages (--alternate_every), an adversarial loss with the
Real-ESRGAN U-Net discriminator (--lambda_gan), and an L1 loss between Fourier spectra (--lambda_fft).
train_stage1.sh gives the configuration of the released model.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# datasets under FLASHREFOCUS_DATA, outputs under FLASHREFOCUS_WORK (defaults <repo>/data, <repo>/work; see paths.py)
_REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FLASHREFOCUS_DATA", _REPO / "data"))
WORK_ROOT = Path(os.environ.get("FLASHREFOCUS_WORK", _REPO / "work"))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repository root
import paths  # noqa: E402
paths.add_to_path()

import lpips  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed  # noqa: E402
from diffusers.optimization import get_scheduler  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402

from pisasr import CSDLoss, PiSASR  # noqa: E402
from dpdd_dataset import build_training_dataset  # noqa: E402
from ram_plus_tagger import RamPlusTagger  # noqa: E402
from train_options import parse_args as parse_data_args  # noqa: E402

OFFICIAL_NEG = ("painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, "
                "unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, "
                "jpeg artifacts, deformed, lowres, over-smooth")
DEFOCUS_NEG = OFFICIAL_NEG.replace(", over-smooth", "") + ", bokeh, shallow depth of field, out of focus, defocus blur"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, default=HERE / "runs/train_deblur")
    p.add_argument("--sd21", default=str(paths.SD21))
    p.add_argument("--init_pkl", default=str(paths.PISA_SR / "preset/models/PiSA-SR/pisa_sr.pkl"),
                   help="initial LoRAs ('' = random LoRA init)")
    p.add_argument("--data_config", default=str(HERE / "configs/stage1_data.yaml"))
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--max_train_steps", type=int, default=100000)
    p.add_argument("--pix_steps", type=int, default=4000)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--lambda_l2", type=float, default=1.0)
    p.add_argument("--lambda_lpips", type=float, default=2.0)
    p.add_argument("--lambda_csd", type=float, default=1.0)
    p.add_argument("--cfg_csd", type=float, default=7.5)
    p.add_argument("--min_dm_step_ratio", type=float, default=0.02)
    p.add_argument("--max_dm_step_ratio", type=float, default=0.5)
    p.add_argument("--neg_prompt_csd", default=DEFOCUS_NEG)
    p.add_argument("--pos_prompt_csd", default="")
    p.add_argument("--null_text_ratio", type=float, default=0.5)
    p.add_argument("--lora_rank_unet_pix", type=int, default=4)
    p.add_argument("--lora_rank_unet_sem", type=int, default=4)
    p.add_argument("--vae_lora_rank", type=int, default=0, help="LoRA rank on the VAE encoder (0 = frozen VAE)")
    p.add_argument("--train_conv_in", action="store_true", help="train the UNet conv_in weights (pixel stage)")
    p.add_argument("--pix_lora_scaling", type=float, default=None, help="scaling of the pixel LoRA (default: peft 8 / r)")
    p.add_argument("--pix_lora_a_std", type=float, default=None, help="std of the pixel LoRA A init (default: peft 1 / r)")
    p.add_argument("--sem_lora_scaling", type=float, default=None, help="as --pix_lora_scaling, for the semantic LoRA")
    p.add_argument("--sem_lora_a_std", type=float, default=None, help="as --pix_lora_a_std, for the semantic LoRA")
    p.add_argument("--lambda_gan", type=float, default=0.0, help="adversarial loss weight (semantic stage)")
    p.add_argument("--lr_disc", type=float, default=1e-4)
    p.add_argument("--lambda_fft", type=float, default=0.0, help="Fourier L1 loss weight (semantic stage)")
    p.add_argument("--timesteps1", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--checkpoint_every", type=int, default=2500)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--resume", default=None, help="checkpoint .pkl to resume from (step read from its name)")
    p.add_argument("--max_steps_this_run", type=int, default=None, help="stop this run after this many steps")
    p.add_argument("--alternate_every", type=int, default=0,
                   help="alternate pixel/semantic stages every N steps (semantic iff (step // N) is odd); 0 = off")
    return p.parse_args()


class UNetDiscriminatorSN(torch.nn.Module):
    """U-Net discriminator with spectral normalisation (Real-ESRGAN, Wang et al. 2021; BSD-3-Clause).
    Input in [0, 1]; output: per-pixel real/fake logits."""

    def __init__(self, num_in_ch=3, num_feat=64):
        super().__init__()
        sn = torch.nn.utils.spectral_norm
        conv = torch.nn.Conv2d
        self.conv0 = conv(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = sn(conv(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = sn(conv(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = sn(conv(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = sn(conv(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = sn(conv(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = sn(conv(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = sn(conv(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = sn(conv(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = conv(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        act = lambda t: F.leaky_relu(t, 0.2)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        x0 = act(self.conv0(x))
        x1 = act(self.conv1(x0))
        x2 = act(self.conv2(x1))
        x3 = act(self.conv3(x2))
        x4 = act(self.conv4(up(x3))) + x2
        x5 = act(self.conv5(up(x4))) + x1
        x6 = act(self.conv6(up(x5))) + x0
        return self.conv9(act(self.conv8(act(self.conv7(x6)))))


def fft_loss(pred, target):
    """Mean |F(pred) - F(target)| of the orthonormal 2-D spectra (same scale as a pixel L1)."""
    return (torch.fft.rfft2(pred.float(), norm="ortho") - torch.fft.rfft2(target.float(), norm="ortho")).abs().mean()


VAE_PATTERNS = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]


def vae_encoder_lora_modules(vae):
    """Every conv / attention projection of the VAE encoder (as MagicBokeh's vae_lora_encoder_modules)."""
    mods = []
    for n, _ in vae.named_parameters():
        if n.startswith("encoder.") and n.endswith(".weight") and any(t in n for t in VAE_PATTERNS) and "norm" not in n:
            mods.append(n[: -len(".weight")])
    return sorted(set(mods))


OFFICIAL_ALPHA = 8.0  # peft default lora_alpha, used by the base code


def lora_scaling(unet, pname):
    """(scaling, rank) of the LoRA layer that owns parameter `pname` (...lora_B.<adapter>.weight)."""
    mod_path, rest = pname.split(".lora_", 1)
    adapter = rest.split(".")[1]
    mod = unet.get_submodule(mod_path)
    return mod.scaling[adapter], mod.r[adapter]


def load_unet_lora(unet, sd_unet):
    """Copy LoRA weights stored in the official convention (update = 8 / r_saved * B A). A smaller saved
    rank is embedded exactly into the first ranks of a larger LoRA; B is converted to this model's scaling."""
    n_copied = n_embedded = 0
    for n, p in unet.named_parameters():
        if "lora" not in n or n not in sd_unet:
            continue
        w = sd_unet[n].to(p.dtype)
        if "lora_A" in n:
            p.data[: w.shape[0]].copy_(w)
        else:  # lora_B: [out, r, ...]
            s_model, _ = lora_scaling(unet, n)
            p.data.zero_()
            p.data[:, : w.shape[1]].copy_(w * (OFFICIAL_ALPHA / w.shape[1]) / s_model)
        if w.shape == p.shape:
            n_copied += 1
        else:
            n_embedded += 1
    return n_copied, n_embedded


def configure_lora(unet, args):
    """Optional pixel / semantic LoRA scaling and A-init overrides (see the module docstring)."""
    for name, mod in unet.named_modules():
        if not hasattr(mod, "lora_A"):
            continue
        for adapter in list(mod.lora_A.keys()):
            kind = "pix" if "pix" in adapter else "sem"
            a_std, scaling = getattr(args, f"{kind}_lora_a_std"), getattr(args, f"{kind}_lora_scaling")
            if a_std:
                torch.nn.init.normal_(mod.lora_A[adapter].weight, std=a_std)
            if scaling:
                mod.scaling[adapter] = scaling


def extra_rank_ratio(unet, r0=4):
    """||B[:, r0:]|| / ||B[:, :r0]|| over the pixel LoRA: how much the ranks beyond the released ones are used."""
    hi = lo = 0.0
    for n, p in unet.named_parameters():
        if "lora_B" in n and "pix" in n and p.shape[1] > r0:
            hi += p.data[:, r0:].float().pow(2).sum().item()
            lo += p.data[:, :r0].float().pow(2).sum().item()
    return (hi / lo) ** 0.5 if lo > 0 else 0.0


def conv_in_params(unet):
    return [p for n, p in unet.named_parameters() if n.startswith("conv_in.") and "lora" not in n]


def vae_lora_params(vae):
    return [p for n, p in vae.named_parameters() if "lora" in n]


def set_pixel_extras(m, trainable, args):
    for p in vae_lora_params(m.vae_fix):
        p.requires_grad = trainable
    for p in conv_in_params(m.unet):
        p.requires_grad = trainable and args.train_conv_in


def save_ckpt(m, path, args):
    """Checkpoint in the base code's format (loadable by its evaluation model) plus the optional VAE-encoder LoRA and conv_in.
    LoRA B matrices are converted to the official scaling convention (8 / r)."""
    m.save_model(path)
    sd = torch.load(path, map_location="cpu")
    for n in list(sd["state_dict_unet"]):
        if "lora_B" in n:
            s_model, r = lora_scaling(m.unet, n)
            sd["state_dict_unet"][n] = sd["state_dict_unet"][n] * (s_model / (OFFICIAL_ALPHA / r))
    if args.vae_lora_rank:
        sd["vae_lora_rank"] = args.vae_lora_rank
        sd["vae_lora_encoder_modules"] = vae_encoder_lora_modules_cache
        sd["state_dict_vae"] = {k: v.cpu() for k, v in m.vae_fix.state_dict().items() if "lora" in k}
    if args.train_conv_in:
        sd["conv_in"] = {k: v.detach().cpu() for k, v in m.unet.state_dict().items()
                         if k.startswith("conv_in.") and "lora" not in k}
    torch.save(sd, path)


vae_encoder_lora_modules_cache = None


def val_pairs():
    """Fixed validation crops (768x768, centre) from the DPDD validation split."""
    root = paths.DATA / "dd_dp_dataset_png/val_c"
    src = sorted((root / "source").glob("*.png"))[:6]
    tgt = sorted((root / "target").glob("*.png"))[:6]
    return list(zip(src, tgt))


def val_pairs_lfdof():
    """Held-out LFDOF test scenes for validation (full 1008x688 images, the middle focal slice)."""
    root = (DATA_ROOT / "LFDOF/test_data")
    scenes = sorted(p for p in (root / "input").iterdir() if p.is_dir())[::8][:6]
    return [(sorted(s.glob("*.png"))[len(list(s.glob("*.png"))) // 2], root / "ground_truth" / f"{s.name}.png")
            for s in scenes]


def lapvar(x):
    """Laplacian variance of the grey version of a [-1, 1] image tensor [1, 3, H, W]."""
    g = (x[0] * 0.5 + 0.5).clamp(0, 1)
    g = (0.299 * g[0] + 0.587 * g[1] + 0.114 * g[2])[None, None] * 255
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device)[None, None]
    return F.conv2d(g.float(), k).var().item()


def collate_pair(samples):
    """The four sources return different extra keys; only the blurred/sharp pair is used here."""
    return {"source": torch.stack([s["source"] for s in samples]), "target": torch.stack([s["target"] for s in samples])}


def center_crop(img, size=768):
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return img.crop((l, t, l + size, t + size))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    acc = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="fp16",
                      project_config=ProjectConfiguration(project_dir=str(args.output_dir)),
                      kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    set_seed(args.seed + acc.process_index)
    main_proc = acc.is_main_process
    if main_proc:
        (args.output_dir / "checkpoints").mkdir(exist_ok=True)
        (args.output_dir / "val").mkdir(exist_ok=True)
        json.dump({k: str(v) for k, v in vars(args).items()}, open(args.output_dir / "args.json", "w"), indent=1)
        tb = SummaryWriter(str(args.output_dir / "tb"))

    # ---- model; optionally start from the pretrained LoRAs
    pargs = SimpleNamespace(pretrained_model_path=args.sd21, pretrained_model_path_csd=args.sd21,
                            lora_rank_unet_pix=args.lora_rank_unet_pix, lora_rank_unet_sem=args.lora_rank_unet_sem,
                            resume_ckpt=None, timesteps1=args.timesteps1, enable_xformers_memory_efficient_attention=False,
                            min_dm_step_ratio=args.min_dm_step_ratio, max_dm_step_ratio=args.max_dm_step_ratio,
                            cfg_csd=args.cfg_csd, null_text_ratio=args.null_text_ratio)
    net = PiSASR(pargs)
    global vae_encoder_lora_modules_cache
    if args.vae_lora_rank:
        from peft import LoraConfig
        vae_encoder_lora_modules_cache = vae_encoder_lora_modules(net.vae_fix)
        net.vae_fix.add_adapter(LoraConfig(r=args.vae_lora_rank, init_lora_weights="gaussian",
                                           target_modules=vae_encoder_lora_modules_cache), adapter_name="vae_enc")
    configure_lora(net.unet, args)
    start_step = 0
    init = args.resume or args.init_pkl
    if init:
        sd = torch.load(init, map_location="cpu")
        missing = [n for n, p in net.unet.named_parameters() if "lora" in n and n not in sd["state_dict_unet"]]
        n_copied, n_embedded = load_unet_lora(net.unet, sd["state_dict_unet"])
        if "state_dict_vae" in sd and args.vae_lora_rank:
            for n, p in net.vae_fix.named_parameters():
                if n in sd["state_dict_vae"]:
                    p.data.copy_(sd["state_dict_vae"][n])
        if "conv_in" in sd:
            net.unet.load_state_dict(sd["conv_in"], strict=False)
        if main_proc:
            print(f"loaded LoRAs from {init} (copied {n_copied}, rank-embedded {n_embedded}, missing {len(missing)}; "
                  f"vae lora {'state_dict_vae' in sd}, conv_in {'conv_in' in sd})")
        if args.resume:
            start_step = int(Path(args.resume).stem.split("_")[-1])
    net_csd = CSDLoss(args=pargs, accelerator=acc)
    net_csd.requires_grad_(False)
    net_lpips = lpips.LPIPS(net="vgg").to(acc.device).eval()
    net_lpips.requires_grad_(False)
    tagger = RamPlusTagger(checkpoint_path=str(paths.RAM_PLUS),
                           image_size=384, vit="swin_l", fallback_prompt="a sharp, clear image", prompt_threshold=0.7,
                           max_tags=50).to(acc.device).eval()
    tagger.requires_grad_(False)
    disc = opt_d = None
    if args.lambda_gan > 0:
        # fp32, outside accelerate's autocast / grad scaler; DDP-wrapped by hand on several GPUs
        disc = UNetDiscriminatorSN().to(acc.device)
        disc_ckpt = Path(args.resume).with_name(f"disc_{start_step}.pt") if args.resume else None
        if disc_ckpt is not None and disc_ckpt.exists():
            disc.load_state_dict(torch.load(disc_ckpt, map_location="cpu"))
            if main_proc:
                print(f"loaded discriminator from {disc_ckpt}")
        if acc.num_processes > 1:
            disc = torch.nn.parallel.DistributedDataParallel(disc, device_ids=[acc.local_process_index])
        opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_disc, betas=(0.9, 0.99))

    def semantic_at(s):
        return (s // args.alternate_every) % 2 == 1 if args.alternate_every else s >= args.pix_steps

    in_sem = semantic_at(start_step)
    adapters = ["default_encoder_pix", "default_decoder_pix", "default_others_pix"]
    if in_sem:
        adapters += ["default_encoder_sem", "default_decoder_sem", "default_others_sem"]
    net.unet.set_adapter(adapters)
    params = [p for n, p in net.unet.named_parameters() if "lora" in n] + vae_lora_params(net.vae_fix)
    if args.train_conv_in:
        params += conv_in_params(net.unet)
    # every parameter that is ever trained must require grad when DDP builds its reducer (set below by stage)
    for p in params:
        p.requires_grad = True
    opt = torch.optim.AdamW(params, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    sched = get_scheduler("constant_with_warmup", optimizer=opt, num_warmup_steps=args.lr_warmup_steps * acc.num_processes)

    # ---- data: DPDD, RealBokeh, LFDOF and synthetic DIV2K mixture (common/dpdd_dataset.py)
    dargs = parse_data_args(["--config", args.data_config])
    dataset = build_training_dataset(dargs, split="train", image_size=args.resolution, use_augmentation=True)
    dl = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                     num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                                     collate_fn=collate_pair)
    net, opt, dl, sched = acc.prepare(net, opt, dl, sched)
    # optimizer / lr-schedule / grad-scaler state saved next to the checkpoint (opt_<step>.pt), so a resume keeps the
    # Adam moments and does not warm up again
    opt_ckpt = Path(args.resume).with_name(f"opt_{start_step}.pt") if args.resume else None
    if opt_ckpt is not None and opt_ckpt.exists():
        st = torch.load(opt_ckpt, map_location="cpu")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        if acc.scaler is not None and st.get("scaler") is not None:
            acc.scaler.load_state_dict(st["scaler"])
        if main_proc:
            print(f"loaded optimizer state from {opt_ckpt}")
    elif main_proc and args.resume:
        print(f"no optimizer state next to {args.resume}; Adam and the lr warm-up start fresh")
    m = acc.unwrap_model(net)
    m.set_train_sem() if in_sem else m.set_train_pix()
    set_pixel_extras(m, not in_sem, args)
    if main_proc:
        n_tr = sum(p.numel() for p in params if p.requires_grad)
        print(f"trainable now {n_tr / 1e6:.2f}M of {sum(p.numel() for p in params) / 1e6:.2f}M; pix rank "
              f"{args.lora_rank_unet_pix}, vae lora {args.vae_lora_rank}, conv_in {args.train_conv_in}")
    if main_proc:
        print(f"dataset {type(dataset).__name__} {len(dataset)} samples; effective batch "
              f"{args.train_batch_size * args.gradient_accumulation_steps * acc.num_processes}; start step {start_step}")

    vals = {"val": val_pairs(), "val_lfdof": val_pairs_lfdof()}
    lambdas = (args.lambda_l2, args.lambda_lpips, args.lambda_csd) if in_sem else (args.lambda_l2, 0.0, 0.0)
    step = start_step
    t_last = time.time()
    done = False
    while not done:
        for batch in dl:
            with acc.accumulate(net):
                x_src = batch["source"] * 2 - 1
                x_tgt = batch["target"] * 2 - 1
                with torch.no_grad():
                    tags = tagger.predict_batch(batch["target"].float()).prompt_texts
                b = {"prompt": [f"{t}, {args.pos_prompt_csd}" if args.pos_prompt_csd else t for t in tags],
                     "neg_prompt": [args.neg_prompt_csd] * x_src.shape[0], "null_prompt": [""] * x_src.shape[0]}
                pred, lat, pe, npe = net(x_src, x_tgt, batch=b, args=pargs)
                l_l2 = F.mse_loss(pred.float(), x_tgt.float()) * lambdas[0]
                l_lp = net_lpips(pred.float(), x_tgt.float()).mean() * lambdas[1] if lambdas[1] > 0 else torch.zeros((), device=pred.device)
                l_csd = net_csd.cal_csd(lat, pe, npe, pargs) * lambdas[2] if lambdas[2] > 0 else torch.zeros((), device=pred.device)
                zero = torch.zeros((), device=pred.device)
                l_fft = fft_loss(pred, x_tgt) * args.lambda_fft if in_sem and args.lambda_fft > 0 else zero
                use_gan = in_sem and disc is not None
                l_g = F.softplus(-disc(pred.float() * 0.5 + 0.5)).mean() * args.lambda_gan if use_gan else zero
                loss = l_l2 + l_lp + l_csd + l_fft + l_g
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(params, args.max_grad_norm)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                if use_gan:  # discriminator step on (target, detached output); drops the D grads of the G loss
                    opt_d.zero_grad(set_to_none=True)
                    logits = disc(torch.cat([x_tgt.float(), pred.detach().float()]) * 0.5 + 0.5)
                    lg_real, lg_fake = logits.chunk(2)
                    l_d = F.softplus(-lg_real).mean() + F.softplus(lg_fake).mean()
                    l_d.backward()
                    opt_d.step(); opt_d.zero_grad(set_to_none=True)
            if not acc.sync_gradients:
                continue
            step += 1
            # switch between the pixel and the semantic stage
            if semantic_at(step) != in_sem:
                m = acc.unwrap_model(net)
                in_sem = not in_sem
                pix = ["default_encoder_pix", "default_decoder_pix", "default_others_pix"]
                m.unet.set_adapter(pix + ["default_encoder_sem", "default_decoder_sem", "default_others_sem"] if in_sem else pix)
                m.set_train_sem() if in_sem else m.set_train_pix()
                set_pixel_extras(m, not in_sem, args)
                lambdas = (args.lambda_l2, args.lambda_lpips, args.lambda_csd) if in_sem else (args.lambda_l2, 0.0, 0.0)
                if main_proc:
                    print(f"step {step}: switched to the {'semantic' if in_sem else 'pixel'} stage", flush=True)
            if main_proc:
                tb.add_scalar("loss/total", loss.item(), step)
                tb.add_scalar("loss/l2", l_l2.item(), step)
                tb.add_scalar("loss/lpips", l_lp.item(), step)
                tb.add_scalar("loss/csd", l_csd.item(), step)
                if args.lambda_fft > 0:
                    tb.add_scalar("loss/fft", l_fft.item(), step)
                if disc is not None and in_sem:
                    tb.add_scalar("loss/gan_g", l_g.item(), step)
                    tb.add_scalar("loss/gan_d", l_d.item(), step)
                    tb.add_scalar("gan/d_real", torch.sigmoid(lg_real).mean().item(), step)
                    tb.add_scalar("gan/d_fake", torch.sigmoid(lg_fake).mean().item(), step)
                tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
                tb.add_scalar("train/stage_semantic", float(in_sem), step)
                if step % 50 == 0:
                    dt = (time.time() - t_last) / 50; t_last = time.time()
                    tb.add_scalar("train/sec_per_step", dt, step)
                    print(f"step {step} loss {loss.item():.4f} l2 {l_l2.item():.4f} lpips {l_lp.item():.4f} "
                          f"csd {l_csd.item():.4f} {dt:.2f}s/step eta {dt * (args.max_train_steps - step) / 3600:.1f}h", flush=True)
                if step % 500 == 0 and args.lora_rank_unet_pix > 4:
                    tb.add_scalar("train/pix_extra_rank_ratio", extra_rank_ratio(acc.unwrap_model(net).unet), step)
                if (step % args.checkpoint_every == 0 or step == args.pix_steps or step == args.max_train_steps
                        or (args.alternate_every and step % args.alternate_every == 0)):
                    save_ckpt(acc.unwrap_model(net), str(args.output_dir / "checkpoints" / f"model_{step}.pkl"), args)
                    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(),
                                "scaler": acc.scaler.state_dict() if acc.scaler is not None else None, "step": step},
                               args.output_dir / "checkpoints" / f"opt_{step}.pt")
                    if disc is not None:
                        d = disc.module if hasattr(disc, "module") else disc
                        torch.save(d.state_dict(), args.output_dir / "checkpoints" / f"disc_{step}.pt")
                if step % args.val_every == 0 or step == 1:
                    validate(acc.unwrap_model(net), vals, net_lpips, tb, step, args)
            acc.wait_for_everyone()
            if step >= args.max_train_steps or (args.max_steps_this_run and step - start_step >= args.max_steps_this_run):
                done = True
                break
    if main_proc:
        tb.close()


@torch.no_grad()
def validate(model, vals, net_lpips, tb, step, args):
    """DPDD val crops ('val') and held-out LFDOF test images ('val_lfdof'): PSNR, LPIPS and sharpness
    (Laplacian variance of the output / of the ground truth; < 1 means blurrier than the ground truth)."""
    model.unet.eval()
    for tag, pairs in vals.items():
        psnrs, lps, sharp, grid = [], [], [], []
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
            mse = F.mse_loss(out * 0.5 + 0.5, y * 0.5 + 0.5).item()
            psnrs.append(10 * math.log10(1 / max(mse, 1e-10)))
            lps.append(net_lpips(out.float(), y.float()).mean().item())
            sharp.append(lapvar(out) / max(lapvar(y), 1e-6))
            if i < 3 and tag == "val":
                grid.append(torch.cat([x, out, y], dim=3)[0] * 0.5 + 0.5)
        tb.add_scalar(f"{tag}/psnr", float(np.mean(psnrs)), step)
        tb.add_scalar(f"{tag}/lpips", float(np.mean(lps)), step)
        tb.add_scalar(f"{tag}/sharpness", float(np.mean(sharp)), step)
        if grid:
            tb.add_image(f"{tag}/input_output_gt", torch.cat(grid, dim=1).clamp(0, 1), step)
            TF.to_pil_image(torch.cat(grid, dim=1).clamp(0, 1).cpu()).resize((1152, 1152)).save(
                args.output_dir / "val" / f"{step:06d}.jpg")
    model.unet.train()


if __name__ == "__main__":
    main()
