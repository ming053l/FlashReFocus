#!/usr/bin/env python
"""Table 6 protocol for the faster pipeline (fast_refocus.py): time per call including image read/write, over --runs calls
after --warmup calls; First = Stage I + depth + render, New focus = render with the cached depth. Also compares every
output with the original code path (cost_pisa.run + to_pil_image; BokehliciousFocusEngine.infer_image)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import fast_refocus as FR


def timed(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sum(ts) / len(ts), torch.cuda.max_memory_allocated() / 2**30


def psnr(a, b):
    a, b = np.asarray(Image.open(a)).astype(np.float64), np.asarray(Image.open(b)).astype(np.float64)
    mse = ((a - b) ** 2).mean()
    return (float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)), float(np.abs(a - b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--refiner", required=True)
    ap.add_argument("--refiner_fp16", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile_mode", default=None, help="e.g. reduce-overhead (CUDA graphs)")
    ap.add_argument("--channels_last", action="store_true")
    ap.add_argument("--no_cudnn_benchmark", action="store_true")
    ap.add_argument("--render_amp", choices=["none", "bf16", "fp16"], default="none")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--real", type=Path, default=Path("/tmp/claude-1016/render_real_1536.png"))
    ap.add_argument("--work", type=Path, default=Path("/tmp/claude-1016/fast_refocus"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    src = a.work / "random_1536.png"
    Image.fromarray(np.random.RandomState(0).randint(0, 255, (1536, 1536, 3), dtype=np.uint8)).save(src)
    aif, bokeh = a.work / "aif.png", a.work / "bokeh.png"

    FR.setup_backend(not a.no_cudnn_benchmark)
    s1 = FR.FastStageI(a.ckpt, a.refiner, refiner_fp16=a.refiner_fp16, compile_=a.compile, channels_last=a.channels_last,
                       compile_mode=a.compile_mode)
    amp = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[a.render_amp]
    rd = FR.FastRenderer(channels_last=a.channels_last, amp_dtype=amp, compile_=a.compile, compile_mode=a.compile_mode)
    res = dict(gpu=torch.cuda.get_device_name(), refiner_fp16=a.refiner_fp16, compile=a.compile, compile_mode=a.compile_mode,
               channels_last=a.channels_last, cudnn_benchmark=not a.no_cudnn_benchmark, render_amp=a.render_amp,
               params_B=(s1.params + rd.params) / 1e9)

    # outputs vs the original code path, on the random and a real image
    import cost_pisa
    for tag, img in (("random", src), ("real", a.real)):
        s1(img, a.work / f"{tag}_aif_fast.png")
        with torch.no_grad():
            x = TF.to_tensor(Image.open(img).convert("RGB")).unsqueeze(0).cuda() * 2 - 1
            y = cost_pisa.run(s1.m, s1.ref, s1.default, x)
        TF.to_pil_image((y[0].float().clamp(-1, 1) * 0.5 + 0.5).cpu()).save(a.work / f"{tag}_aif_orig.png")
        res[f"stage1_{tag}_psnr_maxdiff"] = psnr(a.work / f"{tag}_aif_fast.png", a.work / f"{tag}_aif_orig.png")
        orig_aif = a.work / f"{tag}_aif_orig.png"
        rd.set_image(orig_aif)
        rd.render(768, 768, 0.5, a.work / f"{tag}_bokeh_fast.png")
        rd.eng.infer_image(orig_aif, a.work / f"{tag}_bokeh_orig.png", focus_x=768, focus_y=768, bokeh_embedding=0.5,
                           marked_output=a.work / f"{tag}_marker.png", focus_map_type=rd.bc.focus_map_type)
        res[f"render_{tag}_psnr_maxdiff"] = psnr(a.work / f"{tag}_bokeh_fast.png", a.work / f"{tag}_bokeh_orig.png")

    # timing (Table 6 protocol): every call reads its input and writes its PNG
    t1, m1 = timed(lambda: s1(src, aif), a.warmup, a.runs)
    first = lambda: (rd.set_image(aif), rd.render(768, 768, 0.5, bokeh))
    t2, m2 = timed(first, a.warmup, a.runs)
    pts = [(300, 400), (1200, 900), (768, 200), (500, 1300)]
    k = iter(range(10 ** 9))
    t3, m3 = timed(lambda: rd.render(*pts[next(k) % 4], 0.5, bokeh), a.warmup, a.runs)
    res.update(stage1_s=t1, depth_render_s=t2, first_s=t1 + t2, new_focus_s=t3,
               stage1_peak_GiB=m1, render_peak_GiB=max(m2, m3), peak_GiB=max(m1, m2, m3))
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
