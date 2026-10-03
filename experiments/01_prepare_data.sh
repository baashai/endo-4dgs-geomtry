#!/bin/bash
# Download StereoMIS (Zenodo 7727692, ~10 GB) and build every split and prior used in the paper.
source "$(dirname "$0")/common.sh"
mkdir -p data && cd data
[ -f StereoMIS.zip ] || wget -c -O StereoMIS.zip https://zenodo.org/api/records/7727692/files/StereoMIS.zip/content
echo "f0e413c092b2e63b6c1d81544d25a73c  StereoMIS.zip" | md5sum -c -
unzip -q -n StereoMIS.zip "StereoMIS/P1/*" "StereoMIS/P2_5/*" "StereoMIS/sequences.txt"
cd "$REPO"
# P1, frames 801-1000: rectify (authors' script, writes data/StereoMIS/P1/images_1, images_right_1) and build splits
mkdir -p data/StereoMIS/_run && (cd data/StereoMIS/_run && python ../../../stereomis/process_dataset.py) && rmdir data/StereoMIS/_run
python stereomis/make_split.py --split 1                                    # released-style poses (focal 1039.6 px, 34.2 deg FoV)
python stereomis/make_split.py --split 1 --name split_1f --rectified_focal  # corrected focal (569.0 px, 58.7 deg)
# P2_5, frames 5301-5499, every 2nd frame (static camera)
python stereomis/prepare_sequence.py --seq P2_5 --start 5301 --end 5499 --step 2 --name p2_5 --orig_name p2_5_orig
# Depth-Anything priors (released code path; needs submodules/depth_anything/weights/depth_anything_vitb14.onnx)
for s in split_1 p2_5; do python scripts/pre_dam_dep.py --dataset_root data/stereomis/$s --rgb_paths images; done
ln -sfn ../split_1/depth_dam data/stereomis/split_1f/depth_dam
ln -sfn ../split_1/depth_dam data/stereomis/split_1/dep_png
ln -sfn ../p2_5/depth_dam data/stereomis/p2_5_orig/depth_dam
echo "Now run 01b_moge_priors.sh in the 'moge' environment."
