#!/usr/bin/env python
"""Metrics of the focal-edit consistency study (Table H.2), for FlashReFocus and GenRefocus:
  * in-focus fidelity: PSNR and LPIPS (AlexNet) against the all-in-focus input over pixels near the focal depth
    (|D - D_f| < TAU_FOCUS);
  * consistency: between consecutive focal settings, pixels whose expected blur is unchanged
    (| |D - D_f,i| - |D - D_f,i+1| | < TAU_SAME) should not change; PSNR and mean absolute difference there;
  * blur ratio: out-of-focus sharpness relative to the input (mean |Laplacian| where |D - D_f| > 0.25).
Masks use the renderer depth for both methods (GenRefocus uses Depth Pro internally; the thresholds allow for this).

  focal_edits_metrics.py i    setting (i): FlashReFocus matched to GenRefocus at K = 15 (fidelity from the per-setting
                              match, consistency from the per-scene match), against <scene>/genrefocus
  focal_edits_metrics.py ii   setting (ii): FlashReFocus at 0.5 (<scene>/ours) against <scene>/genrefocus_matched
Writes metrics_setting_<i|ii>.json (per scene and mean) and panel_metrics_setting_<i|ii>.json (the in-focus PSNR and
LPIPS printed under the panels of Figs. H.7 and H.8)."""
import argparse
import json

import lpips
import numpy as np
import torch

from eval_utils import laplacian_sharpness, load_rgb, masked_psnr
from focal_edits_common import AIF, OUT, OUT_OF_FOCUS, TAU_FOCUS, TAU_SAME, scene_dirs

SETTINGS = {   # method -> (folder for in-focus fidelity, folder for consistency)
    "i": {"FlashReFocus": ("ours_matched_per_setting", "ours_matched_per_scene"), "GenRefocus": ("genrefocus",) * 2},
    "ii": {"FlashReFocus": ("ours", "ours"), "GenRefocus": ("genrefocus_matched",) * 2},
}
PANEL_SETTINGS = (0, 2, 5, 7)   # the focal settings shown in the figures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("setting", choices=list(SETTINGS))
    a = ap.parse_args()
    net = lpips.LPIPS(net="alex", spatial=True).eval()     # on the CPU: keeps the GPUs free for timing runs
    lpips_map = lambda x, y: net(torch.from_numpy(x).permute(2, 0, 1)[None] * 2 - 1,
                                 torch.from_numpy(y).permute(2, 0, 1)[None] * 2 - 1)[0, 0].detach().numpy()
    table, panels = {}, {}
    for scene in scene_dirs():
        points = json.loads((scene / "points.json").read_text())
        depth = np.load(scene / "depth.npy")
        aif = load_rgb(AIF / f"{scene.name}.png")
        size = (aif.shape[1], aif.shape[0])
        table[scene.name] = {}
        for method, (fidelity_dir, consistency_dir) in SETTINGS[a.setting].items():
            frames = lambda folder: [load_rgb(scene / folder / f"{i:02d}.png", size) for i in range(len(points))]
            fid_frames = frames(fidelity_dir)
            con_frames = fid_frames if consistency_dir == fidelity_dir else frames(consistency_dir)
            fid_psnr, fid_lpips, con_psnr, con_absdiff, blur = [], [], [], [], []
            for i, p in enumerate(points):
                in_focus = np.abs(depth - p["depth"]) < TAU_FOCUS
                fid_psnr.append(masked_psnr(fid_frames[i], aif, in_focus[..., None].repeat(3, -1)))
                fid_lpips.append(float(lpips_map(fid_frames[i], aif)[in_focus].mean()) if in_focus.sum() >= 100
                                 else float("nan"))
                far = np.abs(depth - p["depth"]) > OUT_OF_FOCUS
                blur.append(laplacian_sharpness(con_frames[i], far) / laplacian_sharpness(aif, far))
                if i + 1 < len(points):
                    same = np.abs(np.abs(depth - p["depth"]) - np.abs(depth - points[i + 1]["depth"])) < TAU_SAME
                    con_psnr.append(masked_psnr(con_frames[i], con_frames[i + 1], same[..., None].repeat(3, -1)))
                    con_absdiff.append(float(np.abs(con_frames[i] - con_frames[i + 1]).mean(-1)[same].mean())
                                       if same.sum() >= 100 else float("nan"))
                if i in PANEL_SETTINGS:
                    panels[f"{scene.name}_{method}_{PANEL_SETTINGS.index(i)}"] = [round(fid_psnr[-1], 2),
                                                                                round(fid_lpips[-1], 3)]
            table[scene.name][method] = dict(infocus_psnr=float(np.nanmean(fid_psnr)),
                                             infocus_lpips=float(np.nanmean(fid_lpips)),
                                             consistency_psnr=float(np.nanmean(con_psnr)),
                                             consistency_absdiff=float(np.nanmean(con_absdiff)),
                                             blur_ratio=float(np.mean(blur)))
        print(scene.name, json.dumps(table[scene.name]), flush=True)
    mean = {m: {k: float(np.nanmean([table[s][m][k] for s in table])) for k in table[next(iter(table))][m]}
            for m in SETTINGS[a.setting]}
    (OUT / f"metrics_setting_{a.setting}.json").write_text(json.dumps({"scenes": table, "mean": mean}, indent=1))
    (OUT / f"panel_metrics_setting_{a.setting}.json").write_text(json.dumps(panels, indent=1))
    print(json.dumps(mean, indent=1))


if __name__ == "__main__":
    main()
