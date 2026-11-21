# Evaluation

Run from `eval/` (`cd eval`). The datasets and evaluation folders are prepared as in
[training/README.md](../training/README.md#data).

| Result | Scripts |
|---|---|
| Defocus deblurring (DPDD, RealDOF; Table 1) | `scripts/deblur.py --pairs annotations/deblurring_pairs/<set>_test.json`, `score_folders.py` |
| Bokeh rendering (EBB! Val176, 3CReal, RealBokeh val.; Tables 3, G.1) | `bokeh_strength_search.py`, `bokeh_table_metrics.py`, `score_folders.py` |
| Strength control (LVCorr) | `lvcorr.py` |
| Consistency across focal edits (Table H.2) | `focal_edits_render_ours.py`, `focal_edits_render_genrefocus.py`, `focal_edits_match_ours.py`, `focal_edits_match_genrefocus.py`, `focal_edits_metrics.py i` / `ii`, `focal_edits_figures.py` |
| Two focal planes (Appendix H) | `two_focal_planes.py` |
| MODEST aperture series (Appendix G) | `build_modest_series.py`, `modest_strength_analysis.py` |
| Effect of the defocus bias on attention (Appendix C.1) | `attention_bias.py` |
| Renderer residual, defocus embedding and strength scale (Appendix C.1) | `renderer_internals.py` |
| Where the detail branch acts (Appendix F) | `detail_branch.py` |
| Depth source: DA-V2 S/B/L and Depth Pro (Appendix G) | `depth_source.py` |
| Held-out model selection of Stage I (Sec. 4.1) | `scripts/deblur_model_selection.py`, `score_folders.py` |
| Inference cost (Tables 2, 6) | `scripts/benchmark_cost.py deblurring` / `refocusing` |

Outputs of compared methods are produced with their own code (under `baselines/`, `FLASHREFOCUS_BASELINES`) and
scored with the same scripts.
