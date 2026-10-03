# Results

| File | Content |
|---|---|
| `results.tsv` | one row per trained run: name, seed, PSNR, SSIM, LPIPS, scale-aligned AbsRel and δ<1.25 vs. SGBM stereo (`scripts/run_stereomis_exp.sh`) |
| `depth_extended.json` | per model: aligned, fixed-scale and metric AbsRel, training-view scale, per-test-frame scale statistics and per-frame values (`stereomis/eval_depth_extended.py`) |
| `priors.json` | Depth-Anything and MoGe-2 priors alone vs. stereo on the test frames (`stereomis/eval_priors.py`) |
| `uncertainty/uncertainty_{p1,p1_lr,p25,p25_lr}.json` | uncertainty benchmark (`stereomis/uncertainty_bench.py`); `_lr` = strict left–right-consistent stereo reference. Signals are computed before any stereo mask is applied. |
| `uncertainty_v1_reference_masked/` | first version, in which the prior alignment used the stereo-valid mask; kept for traceability (superseded) |
| `scales/SCALES.txt` | prior scale factors: `mogeps` ×0.0928 (main experiments), `mogeps2` ×0.1044 (training frames + known-pose filters), `mogesc` stereo-fitted ×0.1645 (P1) and ×0.0572 (P2_5) |
| `scales/p2_5_calib.json` | rectified intrinsics and f·b of the P2_5 clip |

Run names in `results.tsv` follow `experiments/02_train_p1.sh`, `03_pose_refinement.sh` and `04_train_p25.sh`.
