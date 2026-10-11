#!/usr/bin/env python
"""Train the decoder refiner (Sec. 3.1, flashrefocus/deblurring/decoder_refiner.py) on Stage I latents.

Training latents are the frozen Stage I outputs (encoder posterior mode, empty prompt) at the inference detail weight
--detail_weight. Loss: L_ref = MSE + lambda_struct L_struct + lambda_LPIPS^ref LPIPS(VGG), with lambda_struct = 0.2 and
lambda_LPIPS^ref = 0.1; bf16, lr 5e-5 with 500 warm-up steps and cosine decay, 512 crops of the Stage I training
mixture. Trainable: the decoder and the fusion modules (the VAE encoder is frozen).
Validation: held-out DPDD val_c and LFDOF test images, PSNR and sharpness on the full image and LPIPS on the central
768 x 768 crop; the refiner runs on overlapping 1024 tiles (--val_tile) to bound memory. train_decoder_refiner.sh gives
the configuration of the released refiner."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from flashrefocus import paths
from flashrefocus.deblurring.deblurrer import Deblurrer, refine_in_tiles
from flashrefocus.deblurring.decoder_refiner import build_decoder_refiner
from flashrefocus.deblurring.one_step_diffusion import load_one_step_diffusion
from deblurring_data import build_deblurring_dataset

HERE = Path(__file__).resolve().parent
NOT_IN_CHECKPOINT = ("vae.encoder", "vae.quant_conv", "vae.post_quant_conv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1_checkpoint", default=str(paths.STAGE1))
    p.add_argument("--detail_weight", type=float, default=1.25, help="omega of the Stage I latents (inference setting)")
    p.add_argument("--refiner_init", default=str(paths.REFINER),
                   help="initial refiner weights (default: the released refiner); 'none' starts from the SD1.5 decoder")
    p.add_argument("--grad_ckpt", action="store_true",
                   help="recompute decoder up-blocks and fusion modules in backward (same result, less memory)")
    p.add_argument("--output_dir", type=Path, default=paths.WORK / "runs/decoder_refiner")
    p.add_argument("--data_config", default=str(HERE / "configs/deblurring_data.yaml"))
    p.add_argument("--sd15", default=str(paths.SD15))
    p.add_argument("--dpdd_val", default=str(paths.DATA / "dd_dp_dataset_png/val_c"))
    p.add_argument("--lfdof_test", default=str(paths.DATA / "LFDOF/test_data"))
    p.add_argument("--n_val", type=int, default=12, help="images per validation set")
    p.add_argument("--val_tile", type=int, default=1024, help="refiner pixel tile in validation (0 = whole image)")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--max_train_steps", type=int, default=15000)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_struct", type=float, default=0.2)
    p.add_argument("--lambda_lpips", type=float, default=0.1, help="lambda_LPIPS^ref")
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--checkpoint_every", type=int, default=2500)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def collate_pairs(samples):
    return {"source": torch.stack([s["source"] for s in samples]), "target": torch.stack([s["target"] for s in samples])}


def validation_pairs(args):
    dpdd = Path(args.dpdd_val)
    d = list(zip(sorted((dpdd / "source").glob("*.png")), sorted((dpdd / "target").glob("*.png"))))
    lf = Path(args.lfdof_test)
    scenes = sorted(p for p in (lf / "input").iterdir() if p.is_dir())
    l = [(sorted(s.glob("*.png"))[len(list(s.glob("*.png"))) // 2], lf / "ground_truth" / f"{s.name}.png") for s in scenes]
    every = lambda xs: xs[:: max(1, len(xs) // args.n_val)][: args.n_val]
    return {"val_dpdd": every(d), "val_lfdof": every(l)}


def to_unit_clamped(x):
    """[-1, 1] -> [0, 1], clamped as torchmetrics' LPIPS requires."""
    return ((x.float().clamp(-1, 1) + 1) * 0.5).clamp(0, 1)


def laplacian_variance(x):
    g = (x[0].float() * 0.5 + 0.5).clamp(0, 1)
    g = (0.299 * g[0] + 0.587 * g[1] + 0.114 * g[2])[None, None] * 255
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device)[None, None]
    return F.conv2d(g, k).var().item()


class FrozenStage1:
    """Blurred image in [-1, 1] -> restored latent of the frozen Stage I (fp32)."""

    def __init__(self, checkpoint, detail_weight, device):
        if device.index is not None:  # several processes: build Stage I on this process's GPU
            torch.cuda.set_device(device)
        self.deblurrer = Deblurrer(load_one_step_diffusion(checkpoint, detail_weight=detail_weight))
        del self.deblurrer.model.text_encoder, self.deblurrer.model.vae.decoder   # only the encoder and U-Net run
        torch.cuda.empty_cache()

    @torch.no_grad()
    def __call__(self, x):
        return self.deblurrer.restored_latent(x).float()


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


def save_refiner(refiner, path):
    """Every parameter except the VAE encoder and quant convolutions (they come from SD1.5)."""
    torch.save({k: v.detach().cpu() for k, v in refiner.state_dict().items() if not k.startswith(NOT_IN_CHECKPOINT)}, path)


def main():
    args = parse_args()
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    acc = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="bf16")
    torch.manual_seed(args.seed + acc.process_index)
    main_process = acc.is_main_process
    if main_process:
        (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(str(args.output_dir / "tb"))

    stage1 = FrozenStage1(args.stage1_checkpoint, args.detail_weight, acc.device)
    refiner = build_decoder_refiner(args.sd15)
    if args.refiner_init.lower() != "none":
        res = refiner.load_state_dict(torch.load(args.refiner_init, map_location="cpu"), strict=False)
        missing = [k for k in res.missing_keys if not k.startswith(NOT_IN_CHECKPOINT)]
        assert not missing and not res.unexpected_keys, (missing[:5], res.unexpected_keys[:5])
    if args.grad_ckpt:
        for module in list(refiner.vae.decoder.up_blocks) + list(refiner.fusion_blocks):
            forward = module.forward
            module.forward = (lambda f: lambda *a, **k: torch.utils.checkpoint.checkpoint(f, *a, use_reentrant=False, **k)
                              if torch.is_grad_enabled() else f(*a, **k))(forward)
    refiner.vae.requires_grad_(False)
    refiner.vae.decoder.requires_grad_(True)
    params = [p for p in refiner.parameters() if p.requires_grad]
    if main_process:
        print(f"refiner: {sum(p.numel() for p in params) / 1e6:.2f}M trainable")
    opt = torch.optim.AdamW(params, lr=args.learning_rate)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=args.lr_warmup_steps * acc.num_processes,
                          num_training_steps=args.max_train_steps * acc.num_processes)
    lpips_vgg = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True, reduction="mean").to(acc.device).eval()
    lpips_vgg.requires_grad_(False)

    dataset = build_deblurring_dataset(args.data_config, args.resolution)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                                         num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                                         collate_fn=collate_pairs)
    refiner, opt, loader, sched = acc.prepare(refiner, opt, loader, sched)
    refiner.train()
    acc.unwrap_model(refiner).vae.encoder.eval()
    vals = validation_pairs(args)
    if main_process:
        print(f"dataset {len(dataset)}; effective batch {args.train_batch_size * args.gradient_accumulation_steps * acc.num_processes}")

    step, best, t_last, done = 0, -1.0, time.time(), False
    if main_process:
        best = validate(acc.unwrap_model(refiner), stage1, vals, lpips_vgg, tb, 0, args)
    while not done:
        for batch in loader:
            with acc.accumulate(refiner):
                x, y = batch["source"] * 2 - 1, batch["target"] * 2 - 1
                latent = stage1(x)
                with acc.autocast():
                    pred = refiner(x, latent)
                pred = pred.float()
                l_mse = F.mse_loss(pred, y.float()) * args.lambda_mse
                l_struct = struct_loss(pred, y.float()) * args.lambda_struct
                with torch.autocast(device_type="cuda", enabled=False):
                    l_lpips = lpips_vgg(to_unit_clamped(pred), to_unit_clamped(y)) * args.lambda_lpips
                loss = l_mse + l_struct + l_lpips
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if not acc.sync_gradients:
                continue
            step += 1
            if main_process:
                for name, value in (("total", loss), ("mse", l_mse), ("struct", l_struct), ("lpips", l_lpips)):
                    tb.add_scalar(f"loss/{name}", value.item(), step)
                tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
                if step % 50 == 0:
                    dt = (time.time() - t_last) / 50; t_last = time.time()
                    print(f"step {step} loss {loss.item():.4f} mse {l_mse.item():.4f} struct {l_struct.item():.4f} "
                          f"lpips {l_lpips.item():.4f} {dt:.2f}s/step eta {dt * (args.max_train_steps - step) / 3600:.1f}h",
                          flush=True)
                if step % args.checkpoint_every == 0 or step == args.max_train_steps:
                    save_refiner(acc.unwrap_model(refiner), args.output_dir / "checkpoints" / f"refiner_{step}.pt")
                if step % args.val_every == 0 or step == args.max_train_steps:
                    psnr = validate(acc.unwrap_model(refiner), stage1, vals, lpips_vgg, tb, step, args)
                    if psnr > best:
                        best = psnr
                        save_refiner(acc.unwrap_model(refiner), args.output_dir / "checkpoints" / "best.pt")
                        tb.add_scalar("val/best_step", step, step)
            acc.wait_for_everyone()
            if step >= args.max_train_steps:
                done = True
                break
    if main_process:
        tb.close()


@torch.no_grad()
def validate(refiner, stage1, vals, lpips_vgg, tb, step, args):
    """Returns the DPDD validation PSNR, by which best.pt is selected."""
    refiner.eval()
    torch.cuda.empty_cache()
    psnr_dpdd = 0.0
    for tag, pairs in vals.items():
        psnrs, lps, sharpness = [], [], []
        for s, t in pairs:
            src, tgt = Image.open(s).convert("RGB"), Image.open(t).convert("RGB")
            w, h = src.size
            src, tgt = src.crop((0, 0, w - w % 8, h - h % 8)), tgt.crop((0, 0, w - w % 8, h - h % 8))
            x = TF.to_tensor(src)[None].cuda() * 2 - 1
            y = TF.to_tensor(tgt)[None].cuda() * 2 - 1
            latent = stage1(x)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = refine_in_tiles(refiner, x, latent, args.val_tile, 64) if args.val_tile > 0 else refiner(x, latent)
            out = out.float().clamp(-1, 1)
            psnrs.append(10 * math.log10(1 / max(F.mse_loss(out * 0.5 + 0.5, y * 0.5 + 0.5).item(), 1e-10)))
            H, W = out.shape[-2:]
            t0, l0 = max(0, (H - 768) // 2), max(0, (W - 768) // 2)
            crop = lambda v: v[:, :, t0:t0 + 768, l0:l0 + 768]
            lps.append(lpips_vgg(to_unit_clamped(crop(out)), to_unit_clamped(crop(y))).item())
            sharpness.append(laplacian_variance(out) / max(laplacian_variance(y), 1e-6))
        tb.add_scalar(f"{tag}/psnr", float(np.mean(psnrs)), step)
        tb.add_scalar(f"{tag}/lpips_vgg", float(np.mean(lps)), step)
        tb.add_scalar(f"{tag}/sharpness", float(np.mean(sharpness)), step)
        print(f"val step {step} {tag}: PSNR {np.mean(psnrs):.3f} LPIPS-vgg {np.mean(lps):.4f} "
              f"sharpness {np.mean(sharpness):.3f}", flush=True)
        if tag == "val_dpdd":
            psnr_dpdd = float(np.mean(psnrs))
    torch.cuda.empty_cache()
    refiner.train()
    refiner.vae.encoder.eval()
    return psnr_dpdd


if __name__ == "__main__":
    main()
