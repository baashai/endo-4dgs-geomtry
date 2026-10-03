#!/bin/bash
# Bundle-adjusted poses (Table IV, "Poses"). Needs the `colmap` binary on PATH (or --colmap /path/to/colmap).
source "$(dirname "$0")/common.sh"
python stereomis/refine_poses.py --split data/stereomis/split_1f --name split_1fr
python stereomis/refine_poses.py --split data/stereomis/split_1f --name split_1frs --work data/stereomis/colmap_split_1fr --smooth 15
python stereomis/estimate_prior_scale.py --root data/stereomis/split_1fr  --prior moge --out mogepsr
python stereomis/estimate_prior_scale.py --root data/stereomis/split_1frs --prior moge --out mogepsrs
bash $R fr_dam            $C 0 6050 split_1fr  dam
bash $R fr_mogepsr        $C 0 6051 split_1fr  mogepsr
bash $R fr_mogepsr_init   $C 0 6052 split_1fr  mogepsr  $INIT_ONLY
bash $R frs_dam           $C 0 6053 split_1frs dam
bash $R frs_mogepsrs_init $C 0 6054 split_1frs mogepsrs $INIT_ONLY
