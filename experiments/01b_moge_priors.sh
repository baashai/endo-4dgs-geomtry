#!/bin/bash
# MoGe-2 priors (run in the 'moge' conda env) and their scales.
source "$(dirname "$0")/common.sh"
python scripts/pre_moge_dep.py --dataset_root data/stereomis/split_1 --fov_x 58.72 --save_points
python scripts/pre_moge_dep.py --dataset_root data/stereomis/p2_5 --fov_x 59.494
for d in depth_moge normal_moge points_moge; do ln -sfn ../split_1/$d data/stereomis/split_1f/$d; done
echo "Now run 01c_scales.sh in the 'ED4DGS' environment."
