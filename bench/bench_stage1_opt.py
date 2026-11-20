"""Stage I end to end under several inference options; compares outputs with the default path (PSNR)."""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashrefocus.deblurring import Deblurrer

size = int(sys.argv[1]) if len(sys.argv) > 1 else 1536
which = sys.argv[2].split(",") if len(sys.argv) > 2 else ["refiner"]
torch.manual_seed(0)
x = torch.nn.functional.avg_pool2d(torch.rand(1, 3, size, size, device="cuda") * 2 - 1, 9, 1, 4)


def run(d, n=3):
    with torch.no_grad():
        for _ in range(2): out = d(x)
        torch.cuda.synchronize(); s = time.perf_counter()
        for _ in range(n): out = d(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - s) / n * 1000, out.float().clamp(-1, 1)


d = Deblurrer.from_pretrained(separate_unets="unet" in which)
t0, ref = run(d)
print(f"default bf16: {t0:.0f} ms", flush=True)
mods = {"refiner": d.refiner, "encoder": d.model.vae.encoder, "unet": d.model.unet, "unet_pixel": getattr(d.model, "unet_pixel", None)}
if "unet" in which: which = which + ["unet_pixel"]
for k in which:
    mods[k].forward = torch.compile(mods[k].forward, dynamic=False)
t = time.perf_counter(); run(d, 1); print(f"compile time {time.perf_counter() - t:.0f} s", flush=True)
t2, o2 = run(d)
print(f"compiled {which}: {t2:.0f} ms, {10 * torch.log10(4 / ((o2 - ref) ** 2).mean()).item():.1f} dB vs default, "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
