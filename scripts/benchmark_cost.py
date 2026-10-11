#!/usr/bin/env python
"""Inference cost (Tables 2 and 6). Mean time over --runs calls after --warmup calls, peak allocated GPU memory.

  benchmark_cost.py deblurring [--refiner] [--detail-weight 1.25] [--flops]   Table 2: Stage I alone, tensors in and out
                                                                              (no disk I/O), at --size (default 1536)
  benchmark_cost.py refocusing                                                 Table 6: PNG in and out; First = Stage I +
                                                                              depth + rendering, New focus = rendering
                                                                              with the depth of the same image cached

FLOPs (--flops): one traced run with torch.utils.flop_counter (one multiply-add = 2 FLOPs, element-wise operations
excluded); they depend only on tensor shapes. The empty-prompt text encoding is included in every Stage I call."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

from flashrefocus import Refocuser, paths
from flashrefocus.deblurring import Deblurrer
from flashrefocus.deblurring.one_step_diffusion import stage1_parameter_count
from flashrefocus.image_io import read_image, save_image


def timed(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), torch.cuda.max_memory_allocated() / 2**30


def deblurring_cost(a):
    deblurrer = Deblurrer.from_pretrained(refiner=paths.REFINER if a.refiner else None, detail_weight=a.detail_weight)
    x = torch.rand(1, 3, a.size[0], a.size[1], device="cuda") * 2 - 1

    @torch.no_grad()
    def restore():
        deblurrer.prompt_embedding = deblurrer.model.empty_prompt_embedding()
        return deblurrer(x)

    res = {"refiner": a.refiner, "detail_weight": a.detail_weight, "size": a.size,
           "params_M": stage1_parameter_count(deblurrer.model, deblurrer.refiner) / 1e6}
    res["time_s"], res["peak_GiB"] = timed(restore, a.warmup, a.runs)
    if a.flops:
        with FlopCounterMode(display=False) as counter:
            restore()
        res["tflops"] = counter.get_total_flops() / 1e12
    return res


def refocusing_cost(a):
    torch.backends.cudnn.benchmark = True
    a.work.mkdir(parents=True, exist_ok=True)
    src, aif, bokeh = a.work / "input.png", a.work / "aif.png", a.work / "bokeh.png"
    h, w = a.size
    Image.fromarray(np.random.RandomState(0).randint(0, 255, (h, w, 3), dtype=np.uint8)).save(src)
    rf = Refocuser.from_pretrained(refiner_dtype=torch.float16, channels_last=True)
    points = [(300, 400), (w - 336, h - 636), (w // 2, 200), (500, h - 236)]
    state = {"scene": None, "edit": 0}

    def stage1():
        save_image(rf.deblur(read_image(src, "cuda")), aif)

    def depth_and_render():
        state["scene"] = rf.prepare(read_image(aif, "cuda"), deblur=False)
        save_image(rf.render(state["scene"], 0.5, state["scene"].focal_depth_at(w // 2, h // 2)), bokeh)

    def new_focus():
        x, y = points[state["edit"] % len(points)]
        state["edit"] += 1
        save_image(rf.render(state["scene"], 0.5, state["scene"].focal_depth_at(x, y)), bokeh)

    t1, m1 = timed(stage1, a.warmup, a.runs)
    t2, m2 = timed(depth_and_render, a.warmup, a.runs)
    t3, m3 = timed(new_focus, a.warmup, a.runs)
    params = stage1_parameter_count(rf.deblurrer.model, rf.deblurrer.refiner) + sum(
        p.numel() for m in (rf.depth_estimator, rf.renderer) for p in m.parameters())
    return {"size": a.size, "params_B": params / 1e9, "stage1_s": t1, "depth_render_s": t2, "first_s": t1 + t2,
            "new_focus_s": t3, "peak_GiB": max(m1, m2, m3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["deblurring", "refocusing"])
    ap.add_argument("--size", type=int, nargs=2, default=[1536, 1536], metavar=("H", "W"))
    ap.add_argument("--refiner", action="store_true", help="deblurring: decode with the refiner")
    ap.add_argument("--detail-weight", type=float, default=1.25)
    ap.add_argument("--flops", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--work", type=Path, default=paths.WORK / "benchmark_cost")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = deblurring_cost(a) if a.what == "deblurring" else refocusing_cost(a)
    res["gpu"] = torch.cuda.get_device_name()
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
