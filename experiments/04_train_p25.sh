#!/bin/bash
# StereoMIS P2_5 (Table VII).
source "$(dirname "$0")/common.sh"
for s in 0 1 2; do
  bash $R p25_orig      $C $s 6100 p2_5_orig dam
  bash $R p25_fdam      $C $s 6101 p2_5      dam
  bash $R p25_finit_raw $C $s 6102 p2_5      moge   $INIT_ONLY   # monocular, unscaled
  bash $R p25_finit_sc  $C $s 6103 p2_5      mogesc $INIT_ONLY   # stereo-assisted scale x0.0572
done
