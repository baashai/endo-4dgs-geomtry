#!/bin/bash
# Shared settings. REPO: patched Endo-4DGS (setup/install_endo4dgs.sh). Activate conda envs yourself:
#   ED4DGS for training/evaluation, moge for MoGe-2 inference (see README).
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/Endo-4DGS}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
R=scripts/run_stereomis_exp.sh          # train + render + image metrics + stereo depth check -> output/stereomis/results.tsv
C=arguments/stereomis.py
N=arguments/stereomis_priornormal.py
INIT_ONLY="--depth_weight 0 --normal_weight 0 --un_dep_weight 0"   # all depth-related terms and image TV off
