#!/bin/bash
# StereoMIS P1: all trained configurations (Tables III, IV). ~16 min per run on an RTX 4060 Ti.
source "$(dirname "$0")/common.sh"
for s in 0 1 2; do
  bash $R dam            $C $s 6030 split_1  dam                 # original (34.2 deg FoV)
  bash $R f_dam          $C $s 6031 split_1f dam                 # + FoV fix
  bash $R f_dam_init     $C $s 6032 split_1f dam     $INIT_ONLY  # + FoV, DA init only
  bash $R f_mogeps       $C $s 6033 split_1f mogeps              # + FoV, MoGe-2 (x0.093) + depth loss
  bash $R f_mogeps_init  $C $s 6034 split_1f mogeps  $INIT_ONLY  # ours (x0.093)
  bash $R f_mogeps2_init $C $s 6035 split_1f mogeps2 $INIT_ONLY  # ours, held-out-clean scale (x0.104)
  bash $R mogesc         $C $s 6036 split_1  mogesc              # stereo-fitted scale, 34.2 deg (ablation)
done
# single-seed ablations
bash $R moge_raw       $C 0 6037 split_1  moge                  # raw metric MoGe-2 (x1)
bash $R f_mogesc       $C 0 6038 split_1f mogesc                # stereo-fitted, corrected FoV (normal-loss reference)
bash $R f_mogeps_dw003 $C 0 6039 split_1f mogeps --depth_weight 0.003
bash $R f_mogeps_dw03  $C 0 6040 split_1f mogeps --depth_weight 0.03
bash $R f_mogesc_pn01  $N 0 6041 split_1f mogesc --prior_normal_weight 0.01
bash $R f_mogesc_pn05  $N 0 6042 split_1f mogesc --prior_normal_weight 0.05
