"""Tilt-shift demo: the renderer is conditioned on an edited depth map (a vertical ramp) instead of the
estimated one, so a horizontal band stays sharp and blur grows above and below it. No retraining."""
import os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
DEF = Path("/ssd1/chingheng/Defocus"); OUT = DEF / "refocus_bench/tiltshift"; OUT.mkdir(exist_ok=True)
AP = DEF / "refocus_bench/aperture_sweep"
# sample -> (sharp band row as fraction of height, ramp slope, strength, focal point for the depth-based baseline)
CFG = {"07": (0.30, 3.0, 0.6, (0.46, 0.30))}
PLATEAU = 0.06  # half-height of the fully sharp band
sys.path.insert(0, str(DEF / "MyAutoRefocus")); sys.path.insert(0, str(DEF / "SOTA/Bokehlicious"))
# ---- Stage I for samples without a restored image
todo = [s for s in CFG if not (OUT / f"{s}_aif.png").exists()]
for s in list(todo):
    if (AP / f"{s}_aif.png").exists():
        Image.open(AP / f"{s}_aif.png").save(OUT / f"{s}_aif.png"); todo.remove(s)
if todo:
    os.chdir(DEF / "MyAutoRefocus")
    from infer_single_sd1 import AutoRefocusSD1Engine, parse_args as pd
    eng = AutoRefocusSD1Engine(pd(["--input", "x.png", "--output", "y.png", "--no_auto_depth"]))
    for s in todo:
        im = Image.open(next((DEF.parent / "Deblurring-Dataset/Refocus_samples").glob(f"{s}.jp*g"))).convert("RGB")
        k = 1024 / max(im.size); im = im.resize((int(im.width * k) // 8 * 8, int(im.height * k) // 8 * 8), Image.LANCZOS)
        im.save(OUT / f"{s}_input.png"); eng.infer_image(OUT / f"{s}_input.png", OUT / f"{s}_aif.png")
    del eng; torch.cuda.empty_cache()
# ---- renderer with the estimated depth vs an edited (tilt-shift) depth
os.chdir(DEF / "SOTA/Bokehlicious")
import infer_single_focus as F
b = F.parse_args(["--input", "x.png", "--output", "y.png", "--focus-x", "0", "--focus-y", "0", "--bokeh-strength", "0.5"])
ren = F.BokehliciousFocusEngine(checkpoint=b.checkpoint, size=b.size, attn_version=b.attn_version, da2_root=b.da2_root,
                                da2_checkpoint=b.da2_checkpoint, da2_encoder=b.da2_encoder, da2_input_size=b.da2_input_size,
                                no_da2_amp=b.no_da2_amp, da2_amp_dtype=b.da2_amp_dtype, device=b.device, amp=b.amp)

def render(src_pil, depth_cpu, fx, fy, k):
    src = F.to_tensor(src_pil).float().clamp(0, 1).unsqueeze(0).to(ren.device)
    fm, dfoc, dval, xy = F.build_focus_inputs(depth_cpu, fx, fy, b.focus_map_type, ren.device, src.dtype)
    with torch.inference_mode():
        pred = ren.model(source=src, bokeh_strength=torch.tensor([[k]], device=ren.device, dtype=src.dtype),
                         focus_map=fm, depth_map=depth_cpu.unsqueeze(0).to(ren.device), disp_focus=dfoc)
    return F.to_pil_image(pred.clamp(0, 1).squeeze(0).float().cpu())

for s, (row, slope, k, (px, py)) in CFG.items():
    src = Image.open(OUT / f"{s}_aif.png").convert("RGB"); W, H = src.size
    est = F.process_da2_depth_like_dataset(F.predict_da2_depth_from_pil(src, ren.da2_model, ren.da2_input_size, device=ren.device,
                                            use_amp=ren.da2_amp, amp_dtype=ren.da2_amp_dtype), src.size)
    render(src, est, px * W, py * H, k).save(OUT / f"{s}_depthbokeh.png")
    y = torch.linspace(0, 1, H).view(H, 1).expand(H, W)
    dy = (y - row).abs().sub(PLATEAU).clamp(min=0) * torch.sign(y - row)
    tilt = (0.5 + slope * dy / 2).clamp(0, 1).unsqueeze(0)          # vertical ramp with a flat sharp band, focal row -> 0.5
    render(src, tilt, W / 2, row * H, k).save(OUT / f"{s}_tiltshift.png")
    Image.fromarray((tilt[0].numpy() * 255).astype(np.uint8)).save(OUT / f"{s}_tiltdepth.png")
    Image.fromarray((np.clip(est[0].numpy(), 0, 1) * 255).astype(np.uint8)).save(OUT / f"{s}_estdepth.png")
    print(s, "done", flush=True)
