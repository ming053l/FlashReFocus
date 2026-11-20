"""Stage I breakdown at one size: VAE encode, U-Net prediction (two passes at omega 1.25), refiner.
python bench/bench_stage1.py [--size 1536] [--opt ...]"""
import argparse, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashrefocus.deblurring import Deblurrer

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=1536)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--compile", default="", choices=["", "static", "dynamic"])
ap.add_argument("--refiner_dtype", default="bf16")
a = ap.parse_args()
dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[a.refiner_dtype]
d = Deblurrer.from_pretrained(refiner_dtype=dt, compile=a.compile or False)
print("attn processor:", type(next(iter(d.model.unet.attn_processors.values()))).__name__,
      "| vae:", type(next(iter(d.model.vae.attn_processors.values()))).__name__, flush=True)
torch.manual_seed(0)
x = torch.rand(1, 3, a.size, a.size, device="cuda") * 2 - 1


def t(fn):
    for _ in range(2): fn()
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(a.iters): out = fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / a.iters * 1000, out


with torch.no_grad():
    sf = d.scaling_factor
    te, z = t(lambda: d.model.vae.encode(x.to(d.model.weight_dtype)).latent_dist.mode() * sf)
    emb = d.prompt_embedding.expand(1, -1, -1)
    tu, eps = t(lambda: d.model.predict_noise(z, emb))
    lat = (z - eps) / sf
    rd = next(d.refiner.parameters()).dtype
    tr, _ = t(lambda: d.refiner(x.to(rd), lat.to(rd)))
    tt, out = t(lambda: d(x))
print(f"{a.size}^2 compile={a.compile or 'off'}: encode {te:.0f} ms, unet {tu:.0f} ms, refiner {tr:.0f} ms, total {tt:.0f} ms, "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
