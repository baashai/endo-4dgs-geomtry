#!/bin/bash
# Evaluations behind Tables V, VI, VIII and Figs. 4-10.
source "$(dirname "$0")/common.sh"
O=output/stereomis
# prior-only accuracy (Table V)
python stereomis/eval_priors.py --out $O/priors.json
# aligned / fixed-scale / metric depth error and scale drift (Table VI); seed 0 of "dam" is $O/dam_s0
M=""
for s in 0 1 2; do
  M="$M orig_s$s=$O/dam_s$s fdam_s$s=$O/f_dam_s$s fmogeps_s$s=$O/f_mogeps_s$s finit_s$s=$O/f_mogeps_init_s$s"
  M="$M p25orig_s$s=$O/p25_orig_s$s p25fdam_s$s=$O/p25_fdam_s$s p25raw_s$s=$O/p25_finit_raw_s$s p25sc_s$s=$O/p25_finit_sc_s$s"
done
python stereomis/eval_depth_extended.py --out $O/depth_extended.json $M
# uncertainty benchmark (Table VIII, Figs. 7-10): standard and strict (left-right consistent) reference
for seq in p1 p25; do for lr in "" "--lr_check"; do
  ex=""; [ $seq = p1 ] && ex="--example 000898"
  python stereomis/uncertainty_bench.py --seq $seq $lr $ex --out $O/uncertainty_v2
done; done
# depth maps of the qualitative figure (median-error seeds)
python stereomis/dump_frame.py --stem 000898 --out $O/frame_000898.npz orig=$O/dam_s1 fdam=$O/f_dam_s2 finit=$O/f_mogeps_init_s0
