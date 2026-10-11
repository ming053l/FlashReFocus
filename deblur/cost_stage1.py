#!/usr/bin/env python
"""Deblurring cost of Stage I (parameters, FLOPs, time, memory) at a fixed input size, default 1680x1120.

Configurations: plain (one U-Net pass, SD2.1 VAE decoder), refiner (the decoder refiner decodes the latent) and
refiner_l125 (omega = 1.25: two U-Net passes). Inference as deblur.py: empty-prompt text encoding, VAE
encoder, U-Net at t = 1, then the decoder or the refiner.

  --flops : FLOPs of one traced run with torch.utils.flop_counter (one multiply-add = 2 FLOPs, element-wise operations
            excluded); counts depend only on tensor shapes.
  --time  : mean time over --runs after --warmup runs and peak allocated memory, excluding disk I/O."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repository root
import paths  # noqa: E402
paths.add_to_path()
H, W = 1120, 1680


from model import Deblurrer, load_refiner, load_stage1, stage1_params  # noqa: E402


def build(ckpt, sd21, sd15, refiner, omega, precision="fp16", device="cuda"):
    """Stage I (+ refiner) as a Deblurrer; refiner is a checkpoint path or None."""
    return Deblurrer(load_stage1(ckpt, sd21, omega=omega, precision=precision),
                     load_refiner(refiner, sd15, device) if refiner else None)


@torch.no_grad()
def run(deblur, x):
    """One restoration as timed in the paper: the empty-prompt text encoding is included in every call."""
    m = deblur.m
    deblur.emb = m.encode_prompt([""]).to(dtype=deblur.wd)
    return deblur(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["plain", "refiner", "refiner_l125"], required=True)
    ap.add_argument("--flops", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--ckpt", default=str(paths.STAGE1))
    ap.add_argument("--refiner", default=None)
    ap.add_argument("--sd21", default=str(paths.SD21))
    ap.add_argument("--sd15", default=str(paths.SD15))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--height", type=int, default=H)
    ap.add_argument("--width", type=int, default=W)
    a = ap.parse_args()
    globals().update(H=a.height, W=a.width)
    omega = 1.25 if a.config == "refiner_l125" else 1.0
    refiner = (a.refiner or str(paths.REFINER)) if a.config != "plain" else None
    res = {"config": a.config, "resolution": f"{W}x{H}"}
    deblur = build(a.ckpt, a.sd21, a.sd15, refiner, omega)
    x = torch.rand(1, 3, H, W, device="cuda") * 2 - 1
    res["params_M"] = stage1_params(deblur.m, deblur.ref) / 1e6
    if a.time:
        for _ in range(a.warmup):
            run(deblur, x)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(a.runs):
            torch.cuda.synchronize(); t0 = time.perf_counter(); run(deblur, x); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        res.update(time_s=sum(ts) / len(ts), peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9,
                   peak_mem_GiB=torch.cuda.max_memory_allocated() / 2**30,
                   gpu=torch.cuda.get_device_name())
    if a.flops:
        with FlopCounterMode(display=False) as fc:
            run(deblur, x)
        res["tflops"] = fc.get_total_flops() / 1e12
    print(json.dumps(res))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
