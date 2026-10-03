#!/bin/bash
# Prior scales (ED4DGS env).
source "$(dirname "$0")/common.sh"
# (a) known-pose triangulation, as used for the main P1 experiments: all frames, RANSAC only -> x0.0928
python stereomis/estimate_prior_scale.py --root data/stereomis/split_1 --prior moge --out mogeps
# (b) held-out-clean: training frames only + known-pose filters -> x0.1044 (review revision)
python stereomis/estimate_prior_scale.py --root data/stereomis/split_1 --prior moge --out mogeps2 \
    --train_only --cheirality --max_epi 2 --max_reproj 2
# sensitivity to the baseline threshold (2/4/6 mm) and to each filter
for mb in 2 4 6; do python stereomis/estimate_prior_scale.py --root data/stereomis/split_1 --prior moge --train_only --cheirality --max_epi 2 --max_reproj 2 --min_baseline $mb --quiet_pool; done
# (c) stereo-assisted: P1 x0.1645 (ablation), P2_5 x0.0572 (static camera)
python stereomis/fit_stereo_scale.py --root data/stereomis/split_1 --prior moge --out mogesc
python stereomis/fit_stereo_scale.py --root data/stereomis/p2_5 --prior moge --out mogesc
for d in depth_mogeps depth_mogeps2 depth_mogesc; do ln -sfn ../split_1/$d data/stereomis/split_1f/$d; done
