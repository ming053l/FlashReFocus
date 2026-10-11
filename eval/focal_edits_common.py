"""Shared parts of the focal-edit consistency study (Appendix H, "Consistency across focal edits", Table H.2).

Layout of the study folder (FOCAL_EDITS_OUT, default <work>/focal_edits), one folder per scene:
  <scene>/points.json, depth.npy       the eight focal points (far -> near) and the renderer depth
  <scene>/ours, genrefocus             FlashReFocus at strength 0.5 and GenRefocus at its default K = 15
  <scene>/ours_matched_per_setting,    FlashReFocus matched to GenRefocus' blur per focal setting / per scene
  <scene>/ours_matched_per_scene         (setting (i))
  <scene>/genrefocus_matched           GenRefocus with K matched to FlashReFocus at 0.5 (setting (ii))
All-in-focus inputs: FOCAL_EDITS_AIF (default <work>/3creal_lvcorr/aif)."""
import os
from pathlib import Path

from flashrefocus import paths

OUT = Path(os.environ.get("FOCAL_EDITS_OUT", str(paths.WORK / "focal_edits")))
AIF = Path(os.environ.get("FOCAL_EDITS_AIF", str(paths.WORK / "3creal_lvcorr/aif")))
TAU_FOCUS = 0.04       # |D - D_f| below this: in focus (in-focus fidelity mask)
TAU_SAME = 0.02        # | |D - D_f,i| - |D - D_f,i+1| | below this: the blur should not change between consecutive edits
OUT_OF_FOCUS = 0.25    # |D - D_f| above this: clearly out of focus (blur matching)


def scene_dirs(out=OUT):
    return sorted(p for p in out.iterdir() if (p / "points.json").exists())
