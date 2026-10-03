#!/bin/bash
# Train + render + image metrics + stereo depth check for one StereoMIS config/seed.
# Usage: bash scripts/run_stereomis_exp.sh <name> <config> [seed] [port] [split] [depth_prior] [extra train.py args...]
# Appends one row to output/stereomis/results.tsv
set -e
name=$1; cfg=$2; seed=${3:-0}; port=${4:-6030}; split=${5:-split_1}; prior=${6:-}
exp="stereomis/${name}_s${seed}"
out="output/${exp}"
mkdir -p output/stereomis
PYTHONPATH='.' python train.py -s "data/stereomis/$split" --port "$port" --expname "$exp" --configs "$cfg" --seed "$seed" \
    ${prior:+--depth_prior "$prior"} "${@:7}" \
    > "output/stereomis/${name}_s${seed}.train.log" 2>&1
python render.py --model_path "$out" --skip_video --skip_train --configs "$cfg" > "output/stereomis/${name}_s${seed}.render.log" 2>&1
python metrics.py --model_path "$out" > /dev/null 2>&1
geo=$(PYTHONPATH='.' python stereomis/eval_depth_stereo.py "$out" 2>/dev/null | grep AbsRel | awk '{print $3"\t"$5}')
img=$(python -c "import json;d=list(json.load(open('$out/results.json')).values())[0];print(f\"{d['PSNR']:.3f}\t{d['SSIM']:.4f}\t{d['LPIPS']:.4f}\")")
[ -f output/stereomis/results.tsv ] || printf "name\tseed\tPSNR\tSSIM\tLPIPS\tAbsRel\tdelta1\n" > output/stereomis/results.tsv
printf "%s\t%s\t%s\t%s\n" "$name" "$seed" "$img" "$geo" | tee -a output/stereomis/results.tsv
