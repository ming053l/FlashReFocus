#!/usr/bin/env python
"""Figures and videos of the focal-edit study (project page): per scene, a strip with the all-in-focus input and four
focal settings far -> near for FlashReFocus and GenRefocus, the difference map between the two middle edits, the
rack-focus and strength-sweep videos, and a side-by-side video of the eight edits with the time per edit."""
import argparse
import json

import numpy as np

from eval_utils import add_label, encode_video, load_rgb, mark_point, save_rgb
from focal_edits_common import AIF, OUT, scene_dirs

STRIP_SETTINGS = (0, 2, 5, 7)
METHODS = {"FlashReFocus": "ours", "GenRefocus": "genrefocus"}


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    meta = json.loads((OUT / "meta_ours.json").read_text())
    genrefocus_meta = json.loads((OUT / "meta_genrefocus.json").read_text())
    page = OUT / "project_page"
    page.mkdir(exist_ok=True)
    for scene in scene_dirs():
        s, info = scene.name, meta["scenes"][scene.name]
        points = info["points"]
        aif = load_rgb(AIF / f"{s}.png")
        size = (aif.shape[1], aif.shape[0])
        frame = lambda folder, i: load_rgb(scene / folder / f"{i:02d}.png", size)
        rows = []
        for name, folder in METHODS.items():
            k = len(points) // 2 - 1
            diff = np.abs(frame(folder, k) - frame(folder, k + 1)).mean(-1)
            save_rgb(np.repeat(np.clip(diff * 4, 0, 1)[..., None], 3, -1), scene / f"diffmap_{name}.png")
            tiles = [add_label(aif, "AiF input") if name == "FlashReFocus" else np.ones_like(aif)]
            for j, i in enumerate(STRIP_SETTINGS):
                tile = mark_point(frame(folder, i), points[i]["x"], points[i]["y"])
                tiles.append(add_label(tile, f"{name}  focus {j + 1}") if j == 0 else tile)
            rows.append(np.concatenate([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=1) for t in tiles], 1))
        save_rgb(np.concatenate(rows, 0), scene / "strip_far_to_near.png")
        encode_video(scene / "video", "%03d.png", page / f"{s}_rack_focus", 24)
        encode_video(scene / "strength", "%03d.png", page / f"{s}_strength_sweep", 15)
        side_by_side = scene / "side_by_side"
        side_by_side.mkdir(exist_ok=True)
        t_ours, t_genrefocus = info["render_s_mean"], float(np.mean(genrefocus_meta["seconds_per_frame"][s]))
        for i, p in enumerate(points):
            a = add_label(mark_point(frame("ours", i), p["x"], p["y"]), f"FlashReFocus ({t_ours:.2f} s / edit)")
            b = add_label(mark_point(frame("genrefocus", i), p["x"], p["y"]), f"GenRefocus ({t_genrefocus:.0f} s / edit)")
            save_rgb(np.concatenate([a, np.ones((a.shape[0], 12, 3)), b], 1), side_by_side / f"{i:02d}.png")
        encode_video(side_by_side, "%02d.png", page / f"{s}_side_by_side", 2)


if __name__ == "__main__":
    main()
