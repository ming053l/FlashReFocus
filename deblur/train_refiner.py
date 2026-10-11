#!/usr/bin/env python
"""Train the decoder refiner on Stage I latents.

The refiner (refiner/vae_refiner_wo_d.py) is a Stable Diffusion 1.5 VAE decoder with fusion modules that inject
multi-scale features of the blurred input and its Laplacian high-frequency map into each decoder block. SD1.5 and SD2.1
share the VAE encoder and quant_conv, so it decodes the latents of the SD2.1-based Stage I. Training latents are the
frozen Stage I one-step outputs z - UNet(z) (encoder posterior mode, empty prompt) at the inference weight
omega = --lambda_sem; --stage1_ckpt with several checkpoints and --lambda_sem_choices draw one per step.

Loss: L_ref = MSE + lambda_struct L_struct + lambda_LPIPS^ref LPIPS(VGG), lambda_struct = 0.2, lambda_LPIPS^ref = 0.1; bf16, lr 5e-5 with 500 warm-up steps and cosine decay, 512 crops of the Stage I
training mixture. Trainable: decoder and fusion modules (the refiner's own encoder is frozen); --freeze_decoder trains
the fusion modules only, --decoder_lora_rank trains LoRA adapters on the decoder instead of its weights.
Validation: held-out DPDD val_c and LFDOF test images, PSNR and sharpness on the full image and LPIPS on the central
768 x 768 crop; the refiner runs on overlapping 1024 tiles (--val_tile) to bound memory."""
import os
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# datasets under FLASHREFOCUS_DATA, outputs under FLASHREFOCUS_WORK (defaults <repo>/data, <repo>/work; see paths.py)
_REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FLASHREFOCUS_DATA", _REPO / "data"))
WORK_ROOT = Path(os.environ.get("FLASHREFOCUS_WORK", _REPO / "work"))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repository root
import paths  # noqa: E402
paths.add_to_path()
from model import tiled_refine  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", required=True, help="Stage-I checkpoint, or a comma-separated list: each step draws one at random "
                   "(latent augmentation); the first one is used for validation")
    p.add_argument("--lambda_sem_choices", default="", help="comma-separated omegas drawn per step (default: --lambda_sem only)")
    p.add_argument("--refiner_init", default=str(paths.REFINER),
                   help="initial refiner weights (default: the released refiner); 'none' starts from the SD1.5 decoder")
    p.add_argument("--decoder_lora_rank", type=int, default=0, help="> 0: freeze the decoder weights and train LoRA adapters of this rank on its convolutions and attention (merged when saving)")
    p.add_argument("--freeze_decoder", action="store_true", help="train only the fusion modules (keep the VAE decoder weights)")
    p.add_argument("--grad_ckpt", action="store_true", help="recompute decoder up-blocks and fusion blocks in backward (same result, less memory)")
    p.add_argument("--lambda_sem", type=float, default=1.0, help="omega of the Stage-I latent (inference setting)")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--data_config", default=str(HERE / "configs/stage1_data.yaml"))
    p.add_argument("--sd21", default=str(paths.SD21))
    p.add_argument("--sd15", default=str(paths.SD15))
    p.add_argument("--dpdd_val", default=str(paths.DATA / "dd_dp_dataset_png/val_c"))
    p.add_argument("--lfdof_test", default=str(DATA_ROOT / "LFDOF/test_data"))
    p.add_argument("--n_val", type=int, default=12, help="images per validation set")
    p.add_argument("--val_tile", type=int, default=1024, help="refiner pixel tile in validation (0 = whole image)")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--max_train_steps", type=int, default=15000)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--mse_weight", type=float, default=1.0)
    p.add_argument("--struct_weight", type=float, default=0.2, help="lambda_struct")
    p.add_argument("--lpips_weight", type=float, default=0.1)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--checkpoint_every", type=int, default=2500)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def collate_pair(samples):
    return {"source": torch.stack([s["source"] for s in samples]), "target": torch.stack([s["target"] for s in samples])}


def val_pairs(args):
    dpdd = Path(args.dpdd_val)
    d = list(zip(sorted((dpdd / "source").glob("*.png")), sorted((dpdd / "target").glob("*.png"))))
    lf = Path(args.lfdof_test)
    scenes = sorted(p for p in (lf / "input").iterdir() if p.is_dir())
    l = [(sorted(s.glob("*.png"))[len(list(s.glob("*.png"))) // 2], lf / "ground_truth" / f"{s.name}.png") for s in scenes]
    step = lambda xs: xs[:: max(1, len(xs) // args.n_val)][: args.n_val]
    return {"val_dpdd": step(d), "val_lfdof": step(l)}


def to01(x):
    """[-1, 1] -> [0, 1], clamped as torchmetrics' LPIPS requires (as in the refiner loss)."""
    return ((x.float().clamp(-1, 1) + 1) * 0.5).clamp(0, 1)


def lapvar(x):
    g = (x[0].float() * 0.5 + 0.5).clamp(0, 1)
    g = (0.299 * g[0] + 0.587 * g[1] + 0.114 * g[2])[None, None] * 255
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device)[None, None]
    return F.conv2d(g, k).var().item()


class Stage1Latents:
    """Frozen Stage I: blurred image in [-1, 1] -> restored (unscaled) latent, as model.Deblurrer.latent."""

    def __init__(self, ckpt, sd21, device, lambda_sem=1.0):
        from model import Deblurrer, load_stage1
        if device.index is not None:  # several processes: build Stage I on this process's GPU
            torch.cuda.set_device(device)
        self.d = Deblurrer(load_stage1(ckpt, sd21, omega=lambda_sem))
        del self.d.m.text_encoder, self.d.m.vae.decoder     # only the empty-prompt embedding and the encoder are used
        torch.cuda.empty_cache()

    @torch.no_grad()
    def __call__(self, x, omega=None):
        if omega is not None:                        # built in two-pass mode; omega = 1 is the single merged pass
            self.d.omega = omega
        return self.d.latent(x).float()


def struct_loss(pred, target, eps=1e-3):
    """L_struct: l1 error after fitting, per image, the affine color map (3x3 matrix and offset) that best maps pred to
    target in the least-squares sense (ridge eps); the fit is not differentiated. Ignores global color shifts and
    penalizes structural errors."""
    b, c, h, w = pred.shape
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred, target = pred.float(), target.float()
        x = torch.cat([pred.permute(0, 2, 3, 1).reshape(b, -1, c), torch.ones((b, h * w, 1), device=pred.device)], -1)
        xtx = torch.bmm(x.transpose(1, 2), x).float()
        xty = torch.bmm(x.transpose(1, 2), target.permute(0, 2, 3, 1).reshape(b, -1, c)).float()
        eye = torch.eye(c + 1, device=pred.device).unsqueeze(0)
        fit = torch.linalg.solve(xtx + eps * eye, xty).transpose(1, 2).detach()
        pred_aug = torch.cat([pred.reshape(b, c, -1), torch.ones((b, 1, h * w), device=pred.device)], dim=1)
        return F.l1_loss(torch.bmm(fit, pred_aug).reshape(b, c, h, w), target)


def main():
    args = parse_args()
    from deblur_data import build_training_dataset
    from refiner.vae_refiner_wo_d import VAEMultiScaleFusionDecoder
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    acc = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="bf16")
    torch.manual_seed(args.seed + acc.process_index)
    main_proc = acc.is_main_process
    if main_proc:
        (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(str(args.output_dir / "tb"))

    omegas = [float(w) for w in args.lambda_sem_choices.split(",") if w] or [args.lambda_sem]
    build_omega = max(omegas + [args.lambda_sem], key=lambda w: w != 1.0)      # two-pass mode if any omega != 1
    stage1_models = [Stage1Latents(c, args.sd21, acc.device, build_omega) for c in args.stage1_ckpt.split(",")]
    stage1_fn = lambda x: stage1_models[0](x, args.lambda_sem)                              # validation: deployed checkpoint and omega
    rng = __import__("random").Random(args.seed + acc.process_index)
    refiner = VAEMultiScaleFusionDecoder(AutoencoderKL.from_pretrained(args.sd15, subfolder="vae"), depth=2,
                                         map_hidden_channels=32, use_pixel_weighting=False, use_high_freq_info=True)
    init = {} if args.refiner_init.lower() == "none" else torch.load(args.refiner_init, map_location="cpu")
    keep = ("vae.encoder", "vae.quant_conv", "vae.post_quant_conv") + (("vae.decoder",) if args.freeze_decoder else ())
    if args.freeze_decoder:                            # keep the original SD VAE decoder; only the fusion modules come from init
        init = {k: v for k, v in init.items() if not k.startswith("vae.decoder")}
    if init:
        res = refiner.load_state_dict(init, strict=False)
        bad = [k for k in res.missing_keys if not k.startswith(keep)]
        assert not bad and not res.unexpected_keys, (bad[:5], res.unexpected_keys[:5])
    if args.grad_ckpt:
        for m in list(refiner.vae.decoder.up_blocks) + list(refiner.fusion_blocks):
            f = m.forward
            m.forward = (lambda f: lambda *a, **k: torch.utils.checkpoint.checkpoint(f, *a, use_reentrant=False, **k)
                         if torch.is_grad_enabled() else f(*a, **k))(f)
    refiner.vae.requires_grad_(False)
    refiner.vae.decoder.requires_grad_(not (args.freeze_decoder or args.decoder_lora_rank > 0))   # train_vae_decoder: true in the original run
    if args.decoder_lora_rank > 0:
        from peft import LoraConfig, inject_adapter_in_model
        inject_adapter_in_model(LoraConfig(r=args.decoder_lora_rank, lora_alpha=args.decoder_lora_rank, init_lora_weights=True,
                                           target_modules=["conv1", "conv2", "conv", "conv_in", "conv_out", "to_q", "to_k", "to_v", "to_out.0"]),
                                refiner.vae.decoder)       # lora_B = 0: training starts from the loaded decoder
    params = [p for p in refiner.parameters() if p.requires_grad]
    if main_proc:
        print(f"refiner: {sum(p.numel() for p in params) / 1e6:.2f}M trainable "
              f"(decoder {sum(p.numel() for p in refiner.vae.decoder.parameters()) / 1e6:.2f}M)")
    opt = torch.optim.AdamW(params, lr=args.learning_rate)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=args.lr_warmup_steps * acc.num_processes,
                          num_training_steps=args.max_train_steps * acc.num_processes)
    lpips_vgg = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True, reduction="mean").to(acc.device).eval()
    lpips_vgg.requires_grad_(False)

    dataset = build_training_dataset(args.data_config, args.resolution)
    dl = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                     num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                                     collate_fn=collate_pair)
    refiner, opt, dl, sched = acc.prepare(refiner, opt, dl, sched)
    refiner.train()
    acc.unwrap_model(refiner).vae.encoder.eval()
    vals = val_pairs(args)
    if main_proc:
        print(f"dataset {len(dataset)}; effective batch {args.train_batch_size * args.gradient_accumulation_steps * acc.num_processes}")

    step, best, t_last, done = 0, -1.0, time.time(), False
    if main_proc:
        best = validate(acc.unwrap_model(refiner), stage1_fn, vals, lpips_vgg, tb, 0, args)
    while not done:
        for batch in dl:
            with acc.accumulate(refiner):
                x = batch["source"] * 2 - 1
                y = batch["target"] * 2 - 1
                lat = rng.choice(stage1_models)(x, rng.choice(omegas))
                with acc.autocast():
                    pred = refiner(x, latent=lat, sample_posterior=False)
                pred = pred.float()
                l_mse = F.mse_loss(pred, y.float()) * args.mse_weight
                l_struct = struct_loss(pred, y.float()) * args.struct_weight
                with torch.autocast(device_type="cuda", enabled=False):
                    l_lp = lpips_vgg(to01(pred), to01(y)) * args.lpips_weight
                loss = l_mse + l_struct + l_lp
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if not acc.sync_gradients:
                continue
            step += 1
            if main_proc:
                tb.add_scalar("loss/total", loss.item(), step)
                tb.add_scalar("loss/mse", l_mse.item(), step)
                tb.add_scalar("loss/struct", l_struct.item(), step)
                tb.add_scalar("loss/lpips", l_lp.item(), step)
                tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
                if step % 50 == 0:
                    dt = (time.time() - t_last) / 50; t_last = time.time()
                    print(f"step {step} loss {loss.item():.4f} mse {l_mse.item():.4f} struct {l_struct.item():.4f} "
                          f"lpips {l_lp.item():.4f} {dt:.2f}s/step eta {dt * (args.max_train_steps - step) / 3600:.1f}h", flush=True)
                if step % args.checkpoint_every == 0 or step == args.max_train_steps:
                    save(acc.unwrap_model(refiner), args.output_dir / "checkpoints" / f"refiner_{step}.pt")
                if step % args.val_every == 0 or step == args.max_train_steps:
                    psnr = validate(acc.unwrap_model(refiner), stage1_fn, vals, lpips_vgg, tb, step, args)
                    if psnr > best:
                        best = psnr
                        save(acc.unwrap_model(refiner), args.output_dir / "checkpoints" / "best.pt")
                        tb.add_scalar("val/best_step", step, step)
            acc.wait_for_everyone()
            if step >= args.max_train_steps:
                done = True
                break
    if main_proc:
        tb.close()


def save(refiner, path):
    """Same format as the original vae_refiner.pt: every non-encoder parameter (decoder + fusion modules); decoder LoRA
    adapters, if any, are merged into the weights."""
    from peft.tuners.lora import LoraLayer
    delta = {n + ".weight": m.get_delta_weight("default") for n, m in refiner.named_modules() if isinstance(m, LoraLayer)}
    sd = {}
    for k, v in refiner.state_dict().items():
        if k.startswith(("vae.encoder", "vae.quant_conv", "vae.post_quant_conv")) or ".lora_" in k:
            continue
        name = k.replace(".base_layer.", ".")
        base = k.rsplit(".base_layer.", 1)[0] + ".weight" if ".base_layer.weight" in k else None
        sd[name] = (v + delta[base].to(v.dtype)).detach().cpu() if base in delta else v.detach().cpu()
    torch.save(sd, path)


@torch.no_grad()
def validate(refiner, stage1_fn, vals, lpips_vgg, tb, step, args):
    refiner.eval()
    torch.cuda.empty_cache()
    psnr_dpdd = 0.0
    for tag, pairs in vals.items():
        ps, lp, sh = [], [], []
        for s, t in pairs:
            src, tgt = Image.open(s).convert("RGB"), Image.open(t).convert("RGB")
            w, h = src.size
            src, tgt = src.crop((0, 0, w - w % 8, h - h % 8)), tgt.crop((0, 0, w - w % 8, h - h % 8))
            x = TF.to_tensor(src)[None].cuda() * 2 - 1
            y = TF.to_tensor(tgt)[None].cuda() * 2 - 1
            lat = stage1_fn(x)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if args.val_tile > 0:
                    out = tiled_refine(refiner, x, lat, args.val_tile, 64)
                else:
                    out = refiner(x, latent=lat, sample_posterior=False)
            out = out.float().clamp(-1, 1)
            ps.append(10 * math.log10(1 / max(F.mse_loss(out * 0.5 + 0.5, y * 0.5 + 0.5).item(), 1e-10)))
            H, W = out.shape[-2:]
            t0, l0 = max(0, (H - 768) // 2), max(0, (W - 768) // 2)
            crop = lambda v: v[:, :, t0:t0 + 768, l0:l0 + 768]
            lp.append(lpips_vgg(to01(crop(out)), to01(crop(y))).item())
            sh.append(lapvar(out) / max(lapvar(y), 1e-6))
        tb.add_scalar(f"{tag}/psnr", float(np.mean(ps)), step)
        tb.add_scalar(f"{tag}/lpips_vgg", float(np.mean(lp)), step)
        tb.add_scalar(f"{tag}/sharpness", float(np.mean(sh)), step)
        print(f"val step {step} {tag}: PSNR {np.mean(ps):.3f} LPIPS-vgg {np.mean(lp):.4f} sharpness {np.mean(sh):.3f}", flush=True)
        if tag == "val_dpdd":
            psnr_dpdd = float(np.mean(ps))
    torch.cuda.empty_cache()
    refiner.train()
    refiner.vae.encoder.eval()
    return psnr_dpdd


if __name__ == "__main__":
    main()
