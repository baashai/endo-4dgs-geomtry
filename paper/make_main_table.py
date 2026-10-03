"""Rows of the main P1 results table from unrounded results (review F27, F30).

Image metrics: results.json of each run; depth: per-frame scale-aligned AbsRel / delta from depth_extended.json
(or results.tsv for runs not in it). Mean +- sample s.d. over seeds (n-1).
  python make_main_table.py  ->  fig/tab_main_rows.tex
"""
import csv
import json
import os
import statistics as st

O = "../Endo-4DGS/output/stereomis/"
E = json.load(open(O + "depth_extended.json"))
TSV = list(csv.DictReader(open(O + "results.tsv"), delimiter="\t"))
ROWS = [  # label, run dirs, depth_extended key (or results.tsv name), hero
    ("Original, reproduced", ["split_1", "dam_s1", "dam_s2"], ("ext", "orig"), False),
    ("+ FoV fix (DA)", [f"f_dam_s{s}" for s in range(3)], ("ext", "fdam"), False),
    ("+ FoV, DA init only$^\\ddagger$", [f"f_dam_init_s{s}" for s in range(3)], ("tsv", "f_dam_init"), False),
    ("+ FoV, MoGe-2 + depth loss", [f"f_mogeps_s{s}" for s in range(3)], ("ext", "fmogeps"), False),
    ("\\textbf{Ours}: FoV, MoGe-2 init ($\\times$0.093)", [f"f_mogeps_init_s{s}" for s in range(3)], ("ext", "finit"), True),
    ("Ours, held-out-clean scale ($\\times$0.104)", [f"f_mogeps2_init_s{s}" for s in range(3)], ("tsv", "f_mogeps2_init"), False),
]


def fmt(v, d):
    m, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
    sds = ("<." + "0" * (d - 1) + "1") if sd < 10 ** -d else ("." + f"{sd:.{d}f}".split(".")[1])
    return f"{m:.{d}f}\\pmsd{{{sds}}}"


lines = []
for label, runs, (src, key), hero in ROWS:
    runs = [r for r in runs if os.path.exists(O + r + "/results.json") and os.path.getsize(O + r + "/results.json") > 0]
    if not runs:
        continue
    img = [list(json.load(open(O + r + "/results.json")).values())[0] for r in runs]
    if src == "ext":
        ks = [k for k in (f"{key}_s{s}" for s in range(3)) if k in E]
        absrel = [E[k]["aligned"] for k in ks]; delta = [E[k]["delta_aligned"] for k in ks]
    else:
        rr = [x for x in TSV if x["name"] == key]
        absrel = [float(x["AbsRel"]) for x in rr]; delta = [float(x["delta1"]) for x in rr]
    cells = [fmt([x["PSNR"] for x in img], 2), fmt([x["SSIM"] for x in img], 4), fmt([x["LPIPS"] for x in img], 3),
             fmt(absrel, 3), fmt(delta, 3)]
    note = "" if len(img) == 3 else f" (n={len(img)})"
    lines.append(("\\rowcolor{ours!12} " if hero else "") + label + note + " & " + " & ".join(cells) + " \\\\")
open("fig/tab_main_rows.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
