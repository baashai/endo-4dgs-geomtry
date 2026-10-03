#!/bin/bash
# Recreate the code used in this work: upstream Endo-4DGS at commit 39c6ce2 + our patch + our new files.
# Usage: bash setup/install_endo4dgs.sh [target_dir]   (default: ./Endo-4DGS)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-$HERE/Endo-4DGS}"
git clone https://github.com/lastbasket/Endo-4DGS.git "$TARGET"
cd "$TARGET"
git checkout 39c6ce2
git submodule update --init submodules/depth_anything submodules/diff-gaussian-rasterization-depth
git apply "$HERE/patches/endo4dgs_39c6ce2.patch"
cp -r "$HERE/overlay/." "$TARGET/"
chmod +x scripts/run_stereomis_exp.sh
echo "Endo-4DGS prepared in $TARGET (patched + overlay). Next: see README, section 'Environment'."
