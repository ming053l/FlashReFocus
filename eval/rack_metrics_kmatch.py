#!/usr/bin/env python
"""Focus-sweep metrics with GenRefocus matched to FlashReFocus: FlashReFocus at its default strength 0.5 (<scene>/ours)
against GenRefocus with K matched to that blur (<scene>/genrefocus_matched, rack_genrefocus_match.py). Same masks and
metrics as rack_analyze.py, plus the out-of-focus sharpness ratio and per-panel in-focus PSNR / LPIPS.
Usage: RACK_OUT=<sweep_dir> RACK_AIF=<aif_dir> rack_metrics_kmatch.py"""
import json

import lpips
import numpy as np
import torch

import rack_analyze as RA
from rack_match_blur import sharp

R, AIF = RA.R, RA.AIF
net = lpips.LPIPS(net="alex", spatial=True).eval()
lp = lambda a, b: net(torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1, torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1)[0, 0].detach().numpy()
meta = json.loads((R / "meta_ours.json").read_text()); km = json.loads((R / "blur_match_genrefocus.json").read_text())
table, panels = {}, {}
for s in meta["scenes"]:
    o = R / s; pts = json.loads((o / "points.json").read_text()); d = np.load(o / "depth.npy"); aif = RA.load(AIF / f"{s}.png")
    size = (aif.shape[1], aif.shape[0]); res = {}
    for name, sub in (("FlashReFocus", "ours"), ("GenRefocus", "genrefocus_matched")):
        fr = [RA.load(o / sub / f"{i:02d}.png", size) for i in range(len(pts))]
        fp, fl, cp, cd, bl = [], [], [], [], []
        for i, p in enumerate(pts):
            m = np.abs(d - p["depth"]) < RA.TAU_FOCUS
            fp.append(RA.mpsnr(fr[i], aif, m[..., None].repeat(3, -1)))
            fl.append(float(lp(fr[i], aif)[m].mean()) if m.sum() >= 100 else float("nan"))
            far = np.abs(d - p["depth"]) > 0.25
            bl.append(sharp(fr[i], far) / sharp(aif, far))
            if i + 1 < len(pts):
                same = np.abs(np.abs(d - p["depth"]) - np.abs(d - pts[i + 1]["depth"])) < RA.TAU_SAME
                cp.append(RA.mpsnr(fr[i], fr[i + 1], same[..., None].repeat(3, -1)))
                cd.append(float(np.abs(fr[i] - fr[i + 1]).mean(-1)[same].mean()) if same.sum() >= 100 else float("nan"))
            if i in (0, 2, 5, 7):
                panels[f"{s}_{'ours' if name == 'FlashReFocus' else 'gen'}_{(0, 2, 5, 7).index(i)}"] = [round(fp[-1], 2), round(fl[-1], 3)]
        res[name] = dict(infocus_psnr=float(np.nanmean(fp)), infocus_lpips=float(np.nanmean(fl)), consistency_psnr=float(np.nanmean(cp)),
                         consistency_absdiff=float(np.nanmean(cd)), blur_ratio=float(np.mean(bl)))
    res["K"] = km[s]["k"]; table[s] = res
    print(s, "K", km[s]["k"], {n: {k: round(v, 4) for k, v in res[n].items()} for n in ("FlashReFocus", "GenRefocus")}, flush=True)
avg = {n: {k: float(np.nanmean([table[s][n][k] for s in table])) for k in table[next(iter(table))][n]} for n in ("FlashReFocus", "GenRefocus")}
(R / "rack_metrics_kmatch.json").write_text(json.dumps({"scenes": table, "mean": avg}, indent=1))
(R / "fig_panel_metrics_kmatch.json").write_text(json.dumps(panels, indent=1))
print(json.dumps(avg, indent=1))
