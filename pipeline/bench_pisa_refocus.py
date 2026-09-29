#!/usr/bin/env python
"""Refocusing cost of the PiSA-SR Stage I pipeline at 1536x1536 (Table 6 protocol, as infer_ours_refocus_pipeline.py).

First = Stage I (PiSA-SR + VAE refiner, omega = lambda_sem = 1.25, full image, no tiling) + Stage II render;
New focus = Stage II render only (Depth Anything V2-L + renderer, reusing the restored AiF image).
Both stages are timed per call including image read/write, as the previous benchmark, over --runs calls after
--warmup warm-up calls; peak memory is the larger of the two stages' peak allocated memory (models loaded once).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

DEFOCUS = Path("/ssd1/chingheng/Defocus")
sys.path.insert(0, str(DEFOCUS / "Stage1_PiSA"))
sys.path.insert(0, str(DEFOCUS / "SOTA" / "Bokehlicious"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cost_pisa  # noqa: E402


def timed(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sum(ts) / len(ts), torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--refiner", required=True)
    ap.add_argument("--lambda_sem", type=float, default=1.25)
    ap.add_argument("--size", type=int, default=1536)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--bokeh_strength", type=float, default=0.5)
    ap.add_argument("--work", type=Path, default=Path("/tmp/claude-1016/refocus_bench_pisa"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sdpa", choices=["none", "da2", "all"], default="none", help="route Stage II attention through SDPA")
    a = ap.parse_args()
    if a.sdpa != "none":
        import sdpa_patch
        sdpa_patch.apply(da2=True, renderer=a.sdpa == "all")
    a.work.mkdir(parents=True, exist_ok=True)
    src, aif, bokeh = a.work / f"random_{a.size}.png", a.work / "aif.png", a.work / "bokeh.png"
    Image.fromarray(np.random.RandomState(0).randint(0, 255, (a.size, a.size, 3), dtype=np.uint8)).save(src)

    # Stage I
    ns = argparse.Namespace(sd21=str(DEFOCUS / "stable-diffusion-2-1-base"), sd15=str(DEFOCUS / "stable-diffusion-v1-5"),
                            myautorefocus=str(DEFOCUS / "MyAutoRefocus"), ckpt=a.ckpt, refiner=a.refiner)
    m, ref, default = cost_pisa.build(ns, a.lambda_sem, True, "cuda", torch.float16)
    p_stage1 = cost_pisa.params(m, ref)

    def restore():
        x = TF.to_tensor(Image.open(src).convert("RGB")).unsqueeze(0).cuda() * 2 - 1
        y = cost_pisa.run(m, ref, default, x)
        TF.to_pil_image((y[0].float().clamp(-1, 1) * 0.5 + 0.5).cpu()).save(aif)

    t1, mem1 = timed(restore, a.warmup, a.runs)
    x = torch.rand(1, 3, a.size, a.size, device="cuda") * 2 - 1
    with cost_pisa.FlopCounterMode(display=False) as fc:
        cost_pisa.run(m, ref, default, x)
    tflops1 = fc.get_total_flops() / 1e12
    del m, ref, x
    torch.cuda.empty_cache()

    # Stage II (same engine and arguments as infer_ours_refocus_pipeline.py)
    from infer_single_focus import BokehliciousFocusEngine, parse_args as parse_bokeh_args
    bc = parse_bokeh_args(["--input", str(aif), "--output", str(bokeh), "--focus-x", "0", "--focus-y", "0",
                           "--bokeh-strength", str(a.bokeh_strength)])
    eng = BokehliciousFocusEngine(checkpoint=bc.checkpoint, size=bc.size, attn_version=bc.attn_version, da2_root=bc.da2_root,
                                  da2_checkpoint=bc.da2_checkpoint, da2_encoder=bc.da2_encoder,
                                  da2_input_size=bc.da2_input_size, no_da2_amp=bc.no_da2_amp,
                                  da2_amp_dtype=bc.da2_amp_dtype, device="cuda", amp=bc.amp)
    p_stage2 = sum(p.numel() for p in eng.da2_model.parameters()) + sum(p.numel() for p in eng.model.parameters())
    render = lambda: eng.infer_image(aif, bokeh, focus_x=a.size / 2, focus_y=a.size / 2, bokeh_embedding=a.bokeh_strength,
                                     marked_output=a.work / "marker.png", focus_map_type=bc.focus_map_type)
    t2, mem2 = timed(render, a.warmup, a.runs)

    res = dict(resolution=f"{a.size}x{a.size}", lambda_sem=a.lambda_sem, sdpa=a.sdpa, ckpt=a.ckpt, refiner=a.refiner,
               gpu=torch.cuda.get_device_name(), stage1_s=t1, render_s=t2, first_s=t1 + t2, new_focus_s=t2,
               stage1_tflops=tflops1, stage1_peak_GiB=mem1, render_peak_GiB=mem2, peak_GiB=max(mem1, mem2),
               params_stage1_B=p_stage1 / 1e9, params_stage2_B=p_stage2 / 1e9, params_B=(p_stage1 + p_stage2) / 1e9)
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
