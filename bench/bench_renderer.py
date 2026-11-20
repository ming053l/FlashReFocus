"""Renderer-only latency and peak memory (warm, synchronized): python bench/bench_renderer.py [--fused] [--compile]"""
import argparse, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashrefocus import paths
from flashrefocus.rendering.renderer import load_renderer

ap = argparse.ArgumentParser()
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
ap.add_argument("--sizes", default="1536x1536,1920x864,3840x1728")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--fused", action="store_true")
ap.add_argument("--compile", action="store_true")
a = ap.parse_args()
dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[a.dtype]
r = load_renderer(paths.RENDERER)
if a.fused:
    from flashrefocus.rendering.fused import enable_fused
    enable_fused(r)
fwd = torch.compile(r, dynamic=False) if a.compile else r
torch.manual_seed(0)
for s in a.sizes.split(","):
    w, h = map(int, s.split("x"))
    img = torch.rand(1, 3, h, w, device="cuda"); dep = torch.rand(1, 1, h, w, device="cuda")
    k = torch.tensor([[1.0]], device="cuda"); fd = torch.tensor([[0.5]], device="cuda")
    def run():
        with torch.no_grad(), torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
            return fwd(img, k, dep, fd)
    for _ in range(3): run()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t = time.perf_counter()
    for _ in range(a.iters): run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / a.iters * 1000
    print(f"{s:>10} {a.dtype} fused={a.fused} compile={a.compile}: {ms:7.1f} ms  peak {(torch.cuda.max_memory_allocated() - base) / 2**30:.2f} GiB", flush=True)
