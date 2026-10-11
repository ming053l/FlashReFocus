"""Train the Stage II renderer in two phases (Sec. 3.2, Appendix C.2; train_renderer.sh has the released settings).

  --phase hypersim    pre-training on synthetic multi-focal Hypersim scenes, targets rendered by BokehMe
  --phase realbokeh   adaptation to RealBokeh pairs from the pre-trained weights (--pretrained), with learning rate
                      --defocus-lr for the defocus-conditioning parameters and --lr for the rest

Loss: lambda_r * L1 + lambda_LPIPS * LPIPS (VGG). A fraction --zero-strength-p of each batch is trained with K = 0 and
the input as target. Checkpoints: <save-dir>/latest.pt every epoch and best.pt by validation PSNR."""
import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import paths  # noqa: E402
from render_data import HypersimPairs, RealBokehPairs  # noqa: E402
from method.model import Renderer, load_state  # noqa: E402

# parameters adapted with --defocus-lr: the defocus fusion, the defocus embedding and strength scale of every
# attention block, and the convolution that returns from the bottleneck
DEFOCUS_PARAMS = ("defocus_fuse", "conv_last", "defocus_head_proj", "aperture_scale_proj")


def build_datasets(args):
    if args.phase == "hypersim":
        return (HypersimPairs(args.data_dir, "train", crop_size=args.crop_size), HypersimPairs(args.data_dir, "val"))
    return (RealBokehPairs(args.data_dir, "train", crop_size=args.crop_size, depth_dir=args.depth_dir),
            RealBokehPairs(args.data_dir, "validation", depth_dir=args.depth_dir))


def build_optimizer(model, args):
    if args.defocus_lr is None:
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    defocus = [p for n, p in model.named_parameters() if any(k in n for k in DEFOCUS_PARAMS)]
    backbone = [p for n, p in model.named_parameters() if not any(k in n for k in DEFOCUS_PARAMS)]
    return torch.optim.Adam([{"params": backbone, "lr": args.lr, "name": "backbone"},
                             {"params": defocus, "lr": args.defocus_lr, "name": "defocus"}])


def unpack(batch, device):
    """I_AiF, target I_bokeh, K, D and D_f of a batch."""
    get = lambda k: batch[k].to(device, non_blocking=True).float()
    return get("source"), get("target"), get("strength"), get("depth"), get("focal_depth")


def zero_strength(image, target, strength, p):
    """With probability p per sample: K = 0 and the input as target (the renderer must then leave the image as is)."""
    if p <= 0:
        return target, strength
    mask = torch.rand(image.size(0), device=image.device) < p
    if not bool(mask.any()):
        return target, strength
    return (torch.where(mask.view(-1, 1, 1, 1), image.detach(), target),
            torch.where(mask.view(-1, 1), torch.zeros_like(strength), strength))


def train_one_epoch(model, loader, optimizer, scaler, l1, lpips, args, device):
    model.train()
    total, step = 0.0, 0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), 1):
        image, target, strength, depth, focal_depth = unpack(batch, device)
        target, strength = zero_strength(image, target, strength, args.zero_strength_p)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=args.amp, dtype=torch.bfloat16):
            out = model(image, strength, depth, focal_depth)
            loss = args.l1_weight * l1(out, target)                                         # lambda_r * L1
            loss = loss + args.lpips_weight * lpips(out.clamp(0.0, 1.0), target.clamp(0.0, 1.0))   # lambda_LPIPS^ren
            lpips.reset()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())
        if args.log_interval > 0 and step % args.log_interval == 0:
            print(f"train step {step}, loss {total / step:.6f}", flush=True)
    return total / max(1, step)


@torch.inference_mode()
def validate(model, loader, lpips_metric, args, device):
    model.eval()
    sums, n = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}, 0
    for batch in tqdm(loader, desc="val", leave=False):
        image, target, strength, depth, focal_depth = unpack(batch, device)
        with autocast(device_type="cuda", enabled=args.amp, dtype=torch.bfloat16):
            out = model(image, strength, depth, focal_depth)
        b = image.size(0)
        sums["psnr"] += psnr(out, target, data_range=1.0).item() * b
        sums["ssim"] += ssim(out, target, data_range=1.0).item() * b
        sums["lpips"] += lpips_metric(out.clamp(0.0, 1.0), target.clamp(0.0, 1.0)).item() * b
        n += b
    return {k: v / max(1, n) for k, v in sums.items()}


def save(path, model, optimizer, scaler, epoch, best_val):
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "best_val": best_val}, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["hypersim", "realbokeh"], required=True)
    ap.add_argument("--data-dir", default=None, help="default: <data>/Hypersim_processed or <data>/RealBokeh")
    ap.add_argument("--depth-dir", default="da2_depth", help="RealBokeh: cached Depth Anything V2 depth "
                                                                  "(generate_da2_depth.py)")
    ap.add_argument("--pretrained", default=None, help="weights to start from (the Hypersim checkpoint)")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4, help="learning rate (of the backbone with --defocus-lr)")
    ap.add_argument("--defocus-lr", type=float, default=None, help="learning rate of DEFOCUS_PARAMS")
    ap.add_argument("--zero-strength-p", type=float, default=0.1)
    ap.add_argument("--l1-weight", type=float, default=1.0)
    ap.add_argument("--lpips-weight", type=float, default=0.2)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--log-interval", type=int, default=50)
    args = ap.parse_args()
    if args.data_dir is None:
        args.data_dir = str(paths.DATA / ("Hypersim_processed" if args.phase == "hypersim" else "RealBokeh"))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    train_set, val_set = build_datasets(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model = Renderer().to(device)
    if args.pretrained:
        load_state(model, args.pretrained)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
    scaler = GradScaler("cuda", enabled=args.amp)
    l1 = nn.L1Loss()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()

    os.makedirs(args.save_dir, exist_ok=True)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, l1, lpips, args, device)
        scheduler.step()
        metrics = validate(model, val_loader, lpips_metric, args, device)
        print(f"epoch {epoch}/{args.epochs}: train loss {loss:.6f}, "
              + ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()), flush=True)
        with open(os.path.join(args.save_dir, "val_metrics.txt"), "a") as f:
            f.write(f"{epoch}," + ",".join(f"{v:.6f}" for v in metrics.values()) + "\n")
        if metrics["psnr"] > best:
            best = metrics["psnr"]
            save(os.path.join(args.save_dir, "best.pt"), model, optimizer, scaler, epoch, best)
        save(os.path.join(args.save_dir, "latest.pt"), model, optimizer, scaler, epoch, best)


if __name__ == "__main__":
    main()
