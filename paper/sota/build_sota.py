"""Build the SOTA comparison table (CSV + LaTeX) for endoscopic dynamic reconstruction.

All third-party numbers are copied from the cited papers' tables (arXiv HTML versions); see
`src` for paper + table. Rows with an empty value were not reported. Run: python build_sota.py
"""
import csv
import subprocess

# columns: group, method, venue, depth input, PSNR, SSIM, LPIPS, AbsRel, FPS, train, GPU mem, GPU, src, note
A = "A"  # StereoMIS P1 frames ~800-1000 (same clip as this work)
B = "B"  # other StereoMIS protocols
C = "C"  # EndoNeRF dataset (cutting / pulling)

E4 = "Endo-4DGS [v3/v4] Tab. 2"
PG = "Endo-PairGS Tab. 2"
G2 = "Endo-G2T Tab. 1"
EW = "EndoWave Tab. II"
US = "this work (3 seeds)"

rows = [
    # ---------------- A: P1 800-1000 ----------------
    (A, "EndoNeRF", "MICCAI'22", "", "21.49", "0.622", "0.360", "", "0.2", "5 h", "4 GB", "RTX 4090", E4, "re-run by Endo-4DGS authors"),
    (A, "EndoSurf", "MICCAI'23", "", "29.87", "0.809", "0.303", "", "0.04", "8 h", "14 GB", "RTX 4090", E4, "re-run by Endo-4DGS authors"),
    (A, "LerPlane-32k", "MICCAI'23", "", "30.80", "0.826", "0.174", "", "1.7", "7 min", "19 GB", "RTX 4090", E4, "re-run by Endo-4DGS authors"),
    (A, "Endo-4DGS", "MICCAI'24", "mono (DA-S)", "32.69", "0.850", "0.148", "", "100", "7 min", "4 GB", "RTX 4090", E4, "v1/v2 of the paper report 31.46 / 0.829 / 0.175"),
    (A, "EndoNeRF", "MICCAI'22", "", "31.12", "0.846", "0.148", "", "", "", "", "A100", PG, "P1 [802,1001]; all retrained"),
    (A, "EndoSurf", "MICCAI'23", "", "31.29", "0.865", "0.168", "", "", "", "", "A100", PG, "P1 [802,1001]; all retrained"),
    (A, "ForPlane-32k", "MedIA", "", "32.19", "0.843", "0.143", "", "", "", "", "A100", PG, "P1 [802,1001]; all retrained"),
    (A, "Endo-4DGS", "MICCAI'24", "mono (DA)", "32.17", "0.850", "0.192", "", "", "", "", "A100", PG, "P1 [802,1001]; all retrained"),
    (A, "EndoGaussian", "arXiv'24", "", "32.79", "0.858", "0.182", "", "", "", "", "A100", PG, "P1 [802,1001]; all retrained"),
    (A, "Deform3DGS", "MICCAI'24", "", "33.35", "0.871", "0.169", "", "", "", "", "A100", PG, "P1 [802,1001]; all retrained"),
    (A, "SurgicalGS", "arXiv'24", "", "32.98", "0.867", "0.172", "", "", "", "", "A100", PG, "P1 [802,1001]; all retrained"),
    (A, "Endo-PairGS", "IJCARS'26", "Splatt3R pairs", "33.56", "0.871", "0.146", "", "100", "~10 min", "4 GB", "A100", PG, "values read via page summary; equal to Endo-TSR Tab. 1"),
    (A, "Endo-TSR", "arXiv'26", "stereo (FoundationStereo)", "38.45", "0.947", "0.074", "", "", "", "", "", "Endo-TSR Tab. 1", "uses stereo depth; baselines copied from Endo-PairGS"),
    (A, "EndoNeRF", "MICCAI'22", "", "30.500", "0.880", "0.078", "", "0.2", "", "", "RTX 4090", G2, "re-run status not stated"),
    (A, "EndoSurf", "MICCAI'23", "", "31.221", "0.890", "0.071", "", "0.04", "", "", "RTX 4090", G2, "re-run status not stated"),
    (A, "Endo-4DGS", "MICCAI'24", "mono (DA)", "32.188", "0.898", "0.066", "", "100", "", "", "RTX 4090", G2, "re-run status not stated"),
    (A, "ST-Endo4DGS", "arXiv'24", "", "32.900", "0.905", "0.060", "", "123", "", "", "RTX 4090", G2, ""),
    (A, "Endo-G2T", "arXiv'25", "", "33.580", "0.914", "0.056", "", "148", "", "", "RTX 4090", G2, "FPS raster-only"),
    (A, "SurgicalGS", "arXiv'24", "", "31.60", "0.854", "0.263", "", "183", "", "", "RTX 4090", EW, "other rows copied from Endo-4DGS"),
    (A, "EndoWave", "arXiv'25", "", "33.26", "0.9126", "0.075", "", "77", "", "", "RTX 4090", EW, "text gives LPIPS 0.0554"),
    (A, "Endo-4DGS, reproduced", "--", "mono (DA-B)", "31.98±0.09", "0.851±0.001", "0.178±0.003", "0.234±0.004", "38", "16.0 min", "0.6 GB*", "RTX 4060 Ti", US, "P1 801-1000; robot poses"),
    (A, "+ corrected FoV (DA)", "--", "mono (DA-B)", "32.33±0.14", "0.854±0.002", "0.186±0.002", "0.235±0.039", "35", "17.0 min", "0.6 GB*", "RTX 4060 Ti", US, ""),
    (A, "+ corrected FoV, DA init only", "--", "mono (DA-B)", "32.62±0.13", "0.860±0.002", "0.165±0.001", "0.376±0.048", "", "15.6 min", "", "RTX 4060 Ti", US, "best PSNR, worst geometry"),
    (A, "+ corrected FoV, MoGe-2 + depth loss", "--", "mono (MoGe-2)", "31.87±0.13", "0.844±0.002", "0.216±0.004", "0.155±0.003", "", "17.1 min", "", "RTX 4060 Ti", US, ""),
    (A, "Ours: corrected FoV + MoGe-2 init", "--", "mono (MoGe-2)", "32.40±0.10", "0.853±0.001", "0.190±0.001", "0.148±0.001", "38", "15.9 min", "0.6 GB*", "RTX 4060 Ti", US, "*render memory"),
    # ---------------- B: other StereoMIS protocols ----------------
    (B, "EndoNeRF", "MICCAI'22", "", "28.86", "0.742", "0.27", "", "0.03", "~6 h", "", "RTX A5000", "Deform3DGS Tab. 1", "3 clips P2_7+P3; SSIM printed in %"),
    (B, "LerPlane", "MICCAI'23", "", "29.46", "0.777", "0.20", "", "1.52", "~10 min", "", "RTX A5000", "Deform3DGS Tab. 1", "3 clips P2_7+P3"),
    (B, "EndoGaussian", "arXiv'24", "", "30.25", "0.828", "0.21", "", "134.5", "151 s", "", "RTX A5000", "Deform3DGS Tab. 1", "3 clips P2_7+P3"),
    (B, "Deform3DGS", "MICCAI'24", "", "30.48", "0.827", "0.21", "", "330.4", "66 s", "", "RTX A5000", "Deform3DGS Tab. 1", "3 clips P2_7+P3"),
    (B, "EndoGaussian", "arXiv'24", "", "30.39", "0.838", "0.21", "", "", "", "", "RTX 3060", "ColorGS Tab. 1", "Deform3DGS protocol"),
    (B, "Deform3DGS", "MICCAI'24", "", "30.68", "0.847", "0.23", "", "", "", "", "RTX 3060", "ColorGS Tab. 1", "Deform3DGS protocol"),
    (B, "ColorGS", "arXiv'25", "", "32.64", "0.896", "0.14", "", "", "", "", "RTX 3060", "ColorGS Tab. 1", "Deform3DGS protocol"),
    (B, "EndoNeRF (P2_7 / P3)", "MICCAI'22", "", "28.87 / 28.33", "0.802 / 0.801", "0.259 / 0.287", "", "0.06", "", "", "V100", "Diff2DGS Tab. II", "RMSE 4.354 / 4.165"),
    (B, "LerPlane (P2_7 / P3)", "MICCAI'23", "", "29.53 / 29.41", "0.784 / 0.781", "0.205 / 0.213", "", "0.97", "", "", "V100", "Diff2DGS Tab. II", "RMSE 8.356 / 7.983"),
    (B, "EndoGaussian (P2_7 / P3)", "arXiv'24", "", "31.11 / 31.03", "0.873 / 0.871", "0.127 / 0.136", "", "90.3", "", "", "V100", "Diff2DGS Tab. II", "RMSE 9.793 / 8.932"),
    (B, "Deform3DGS (P2_7 / P3)", "MICCAI'24", "", "31.83 / 30.91", "0.878 / 0.877", "0.129 / 0.131", "", "214.4", "", "", "V100", "Diff2DGS Tab. II", "RMSE 10.036 / 8.482"),
    (B, "SurgicalGS (P2_7 / P3)", "arXiv'24", "", "31.63 / 31.49", "0.883 / 0.878", "0.152 / 0.165", "", "154.8", "", "", "V100", "Diff2DGS Tab. II", "RMSE 2.747 / 2.536"),
    (B, "Diff2DGS (P2_7 / P3)", "arXiv'26", "", "34.72 / 34.08", "0.912 / 0.903", "0.113 / 0.131", "", "212.3", "", "", "V100", "Diff2DGS Tab. II", "RMSE 3.874 / 3.717"),
    (B, "EndoNeRF (intestine / liver)", "MICCAI'22", "", "28.694 / 27.738", "0.783 / 0.712", "0.279 / 0.345", "", "0.06", "", "13 GB", "RTX 3090", "SurgicalGaussian Tab. 1", "2 fast-motion clips; tool pixels excluded"),
    (B, "EndoSurf (intestine / liver)", "MICCAI'23", "", "29.660 / 28.941", "0.853 / 0.820", "0.204 / 0.248", "", "0.08", "", "14 GB", "RTX 3090", "SurgicalGaussian Tab. 1", ""),
    (B, "LerPlane (intestine / liver)", "MICCAI'23", "", "29.441 / 28.852", "0.822 / 0.793", "0.206 / 0.254", "", "1.45", "", "19 GB", "RTX 3090", "SurgicalGaussian Tab. 1", ""),
    (B, "EndoGaussian (intestine / liver)", "arXiv'24", "", "29.024 / 26.174", "0.805 / 0.728", "0.213 / 0.295", "", "200", "", "2 GB", "RTX 3090", "SurgicalGaussian Tab. 1", ""),
    (B, "SurgicalGaussian (intestine / liver)", "arXiv'24", "stereo (STTR-light)", "31.496 / 31.668", "0.890 / 0.893", "0.145 / 0.135", "", "140", "", "3 GB", "RTX 3090", "SurgicalGaussian Tab. 1", ""),
    (B, "EndoNeRF (intestine / liver)", "MICCAI'22", "", "30.510 / 27.370", "0.833 / 0.680", "0.153 / 0.316", "", "", "", "", "RTX A6000", "SGS Tab. 1", "P2_7 + P2_8 segments"),
    (B, "Deform3DGS (intestine / liver)", "MICCAI'24", "", "32.589 / 32.929", "0.897 / 0.900", "0.131 / 0.110", "", "", "", "", "RTX A6000", "SGS Tab. 1", ""),
    (B, "EndoGS (intestine / liver)", "arXiv'24", "", "35.239 / 34.540", "0.934 / 0.932", "0.060 / 0.051", "", "", "", "", "RTX A6000", "SGS Tab. 1", ""),
    (B, "SGS (intestine / liver)", "WACV'26", "mono (Omnidata)", "34.669 / 34.390", "0.936 / 0.934", "0.048 / 0.047", "", "423", "", "2 GB", "RTX A6000", "SGS Tab. 1", ""),
    (B, "EndoNeRF", "MICCAI'22", "stereo (RAFT)", "28.79", "0.809", "0.266", "0.0315", "0.06", "", "", "RTX 3090", "SurgicalGS Tab. I", "sequences not stated; AbsRel vs RAFT depth"),
    (B, "EndoSurf", "MICCAI'23", "stereo (RAFT)", "29.36", "0.861", "0.211", "0.0189", "0.05", "", "", "RTX 3090", "SurgicalGS Tab. I", ""),
    (B, "LerPlane", "MICCAI'23", "stereo (RAFT)", "29.09", "0.789", "0.179", "0.0354", "0.95", "", "", "RTX 3090", "SurgicalGS Tab. I", ""),
    (B, "EndoGaussian", "arXiv'24", "stereo (RAFT)", "31.02", "0.878", "0.132", "0.0292", "130.2", "", "", "RTX 3090", "SurgicalGS Tab. I", ""),
    (B, "Deform3DGS", "MICCAI'24", "stereo (RAFT)", "31.61", "0.888", "0.135", "0.0330", "308.7", "", "", "RTX 3090", "SurgicalGS Tab. I", ""),
    (B, "SurgicalGS", "arXiv'24", "stereo (RAFT)", "31.54", "0.885", "0.148", "0.0082", "214.1", "", "", "RTX 3090", "SurgicalGS Tab. I", ""),
    (B, "ForPlane (P1_1 / P1_2)", "MedIA", "", "28.47 / 28.95", "0.824 / 0.829", "0.258 / 0.239", "", "1.2", "215 s", "", "RTX 4090", "EH-SurGS Tab. I", "184 / 70 images; all 3000 it., 3 runs"),
    (B, "Deform3DGS (P1_1 / P1_2)", "MICCAI'24", "", "34.54 / 34.87", "0.896 / 0.911", "0.170 / 0.156", "", "357.7", "92 s", "", "RTX 4090", "EH-SurGS Tab. I", ""),
    (B, "Endo-4DGS (P1_1 / P1_2)", "MICCAI'24", "mono (DA)", "33.52 / 34.18", "0.884 / 0.904", "0.180 / 0.150", "", "95.7", "240 s", "", "RTX 4090", "EH-SurGS Tab. I", ""),
    (B, "LGS (P1_1 / P1_2)", "--", "", "23.46 / 25.47", "0.842 / 0.819", "0.270 / 0.315", "", "142.6", "144 s", "", "RTX 4090", "EH-SurGS Tab. I", ""),
    (B, "SurgicalGaussian (P1_1 / P1_2)", "arXiv'24", "", "28.91 / 31.26", "0.831 / 0.859", "0.366 / 0.251", "", "304.0", "159 s", "", "RTX 4090", "EH-SurGS Tab. I", ""),
    (B, "EH-SurGS (P1_1 / P1_2)", "arXiv'25", "", "34.73 / 35.09", "0.898 / 0.913", "0.177 / 0.155", "", "380.3", "111 s", "", "RTX 4090", "EH-SurGS Tab. I", ""),
    (B, "Deform3DGS (P1_1 / P1_2)", "MICCAI'24", "", "28.55 / 32.28", "0.846 / 0.887", "", "", "", "", "", "RTX 4090", "Endo-4DGX Tab. 2", "P1 16400-16697 / 7379-7499; uneven illumination"),
    (B, "Endo-4DGS (P1_1 / P1_2)", "MICCAI'24", "mono (DA)", "26.98 / 32.48", "0.817 / 0.890", "", "", "", "", "", "RTX 4090", "Endo-4DGX Tab. 2", ""),
    (B, "EndoGaussian (P1_1 / P1_2)", "arXiv'24", "", "27.61 / 32.20", "0.841 / 0.896", "", "", "", "", "", "RTX 4090", "Endo-4DGX Tab. 2", ""),
    (B, "Endo-4DGX (P1_1 / P1_2)", "MICCAI'25", "", "28.73 / 32.70", "0.853 / 0.899", "", "", "61", "", "6.5 GB", "RTX 4090", "Endo-4DGX Tab. 2", ""),
    (B, "EndoNeRF (P2-2 / P2-5 / P3)", "MICCAI'22", "", "28.91 / 26.49 / 29.85", "0.703 / 0.688 / 0.751", "0.261 / 0.274 / 0.209", "", "", "", "", "A100", "Endo-PairGS Tab. 2", "all retrained"),
    (B, "EndoSurf (P2-2 / P2-5 / P3)", "MICCAI'23", "", "30.01 / 26.51 / 30.46", "0.780 / 0.738 / 0.811", "0.198 / 0.277 / 0.188", "", "", "", "", "A100", "Endo-PairGS Tab. 2", ""),
    (B, "ForPlane-32k (P2-2 / P2-5 / P3)", "MedIA", "", "28.01 / 28.48 / 31.10", "0.659 / 0.759 / 0.801", "0.286 / 0.170 / 0.139", "", "", "", "", "A100", "Endo-PairGS Tab. 2", ""),
    (B, "Endo-4DGS (P2-2 / P2-5 / P3)", "MICCAI'24", "mono (DA)", "31.45 / 29.06 / 31.21", "0.812 / 0.817 / 0.831", "0.141 / 0.152 / 0.155", "", "", "", "", "A100", "Endo-PairGS Tab. 2", ""),
    (B, "EndoGaussian (P2-2 / P2-5 / P3)", "arXiv'24", "", "29.82 / 28.28 / 31.26", "0.759 / 0.782 / 0.821", "0.169 / 0.173 / 0.166", "", "", "", "", "A100", "Endo-PairGS Tab. 2", ""),
    (B, "Deform3DGS (P2-2 / P2-5 / P3)", "MICCAI'24", "", "30.69 / 29.16 / 31.70", "0.789 / 0.813 / 0.834", "0.144 / 0.147 / 0.160", "", "", "", "", "A100", "Endo-PairGS Tab. 2", ""),
    (B, "SurgicalGS (P2-2 / P2-5 / P3)", "arXiv'24", "", "30.19 / 29.07 / 31.97", "0.775 / 0.805 / 0.841", "0.178 / 0.168 / 0.155", "", "", "", "", "A100", "Endo-PairGS Tab. 2", ""),
    (B, "Endo-PairGS (P2-2 / P2-5 / P3)", "IJCARS'26", "Splatt3R pairs", "32.16 / 29.75 / 32.47", "0.833 / 0.846 / 0.871", "0.100 / 0.103 / 0.110", "", "100", "", "4 GB", "A100", "Endo-PairGS Tab. 2", ""),
    (B, "Endo-TSR (P2-2 / P2-5 / P3)", "arXiv'26", "stereo (FoundationStereo)", "34.83 / 31.99 / 36.19", "0.879 / 0.894 / 0.939", "0.065 / 0.067 / 0.045", "", "", "", "", "", "Endo-TSR Tab. 1", "3 seeds"),
    (B, "EndoNeRF", "MICCAI'22", "", "31.922", "0.857", "0.146", "", "", "", "", "RTX 4090", "Endo-GSMT Tab. 1", "5 segments; SLAM poses; flow-aligned metrics"),
    (B, "LerPlane-32k", "MICCAI'23", "", "31.679", "0.845", "0.113", "", "", "", "", "RTX 4090", "Endo-GSMT Tab. 1", ""),
    (B, "EndoGS", "arXiv'24", "", "32.819", "0.907", "0.099", "", "", "", "", "RTX 4090", "Endo-GSMT Tab. 1", ""),
    (B, "EndoGaussian", "arXiv'24", "", "29.191", "0.827", "0.181", "", "", "", "", "RTX 4090", "Endo-GSMT Tab. 1", ""),
    (B, "Endo-4DGS", "MICCAI'24", "mono (DA)", "31.580", "0.862", "0.124", "", "", "", "", "RTX 4090", "Endo-GSMT Tab. 1", ""),
    (B, "Deform3DGS", "MICCAI'24", "", "32.209", "0.863", "0.124", "", "", "", "", "RTX 4090", "Endo-GSMT Tab. 1", ""),
    (B, "Endo-GSMT", "MICCAI'25", "", "34.703", "0.917", "0.060", "", "", "", "", "RTX 4090", "Endo-GSMT Tab. 1", ""),
    # ---------------- C: EndoNeRF dataset (cutting / pulling) ----------------
    (C, "EndoNeRF", "MICCAI'22", "", "35.84 / 35.43", "0.942 / 0.939", "0.057 / 0.064", "", "0.2", "6 h", "4 GB", "RTX 4090", "Endo-4DGS Tab. 1", ""),
    (C, "EndoSurf", "MICCAI'23", "", "34.89 / 34.91", "0.952 / 0.955", "0.107 / 0.120", "", "0.04", "7 h", "17 GB", "RTX 4090", "Endo-4DGS Tab. 1", ""),
    (C, "LerPlane-32k", "MICCAI'23", "", "34.66 / 31.77", "0.923 / 0.910", "0.071 / 0.071", "", "1.5", "8 min", "20 GB", "RTX 4090", "Endo-4DGS Tab. 1", ""),
    (C, "Endo-4DGS", "MICCAI'24", "mono (DA-S)", "36.56 / 37.85", "0.955 / 0.959", "0.032 / 0.043", "", "100", "4 min", "4 GB", "RTX 4090", "Endo-4DGS Tab. 1", "v1/v2: 36.84 / 37.08"),
    (C, "EndoSurf", "MICCAI'23", "", "34.981 / 35.004", "0.953 / 0.956", "0.106 / 0.120", "", "", "9 h", "", "RTX 3090", "EndoSurf Tab. 1", "RMSE 0.835 / 1.165 mm"),
    (C, "EndoNeRF", "MICCAI'22", "", "34.186 / 34.217", "0.932 / 0.938", "0.151 / 0.160", "", "0.04", "", "15 GB", "RTX 3090", "SurgicalGaussian Tab. 1", ""),
    (C, "LerPlane", "MICCAI'23", "", "35.580 / 36.241", "0.955 / 0.950", "0.101 / 0.102", "", "1.02", "", "20 GB", "RTX 3090", "SurgicalGaussian Tab. 1", ""),
    (C, "EndoGaussian", "arXiv'24", "", "38.287 / 37.308", "0.962 / 0.958", "0.058 / 0.070", "", "190", "", "2 GB", "RTX 3090", "SurgicalGaussian Tab. 1", ""),
    (C, "SurgicalGaussian", "arXiv'24", "", "37.505 / 38.783", "0.961 / 0.970", "0.062 / 0.049", "", "80", "", "4 GB", "RTX 3090", "SurgicalGaussian Tab. 1", ""),
    (C, "Endo-4DGS", "MICCAI'24", "mono (DA)", "36.13 / 37.18", "0.951 / 0.955", "0.054 / 0.072", "", "", "", "", "RTX 4090", "EH-SurGS Tab. I", "3000 it."),
    (C, "Deform3DGS", "MICCAI'24", "", "38.77 / 38.33", "0.967 / 0.961", "0.042 / 0.063", "", "", "", "", "RTX 4090", "EH-SurGS Tab. I", ""),
    (C, "EH-SurGS", "arXiv'25", "", "39.91 / 38.72", "0.972 / 0.963", "0.034 / 0.062", "", "379.7 / 387.0", "105 / 97 s", "", "RTX 4090", "EH-SurGS Tab. I", ""),
    (C, "Endo-4DGS", "MICCAI'24", "mono (DA)", "36.56 / 37.85", "0.955 / 0.959", "0.032 / 0.043", "", "", "", "", "A100", "Endo-PairGS Tab. 3", "identical to Endo-4DGS's own numbers"),
    (C, "Endo-PairGS", "IJCARS'26", "Splatt3R pairs", "38.33 / 38.96", "0.967 / 0.967", "0.020 / 0.022", "", "100", "", "4 GB", "A100", "Endo-PairGS Tab. 3", ""),
    (C, "Endo-TSR", "arXiv'26", "stereo (FoundationStereo)", "40.96 / 39.34", "0.980 / 0.967", "0.019 / 0.035", "", "", "", "", "", "Endo-TSR Tab. 1", ""),
    (C, "Endo-4DGS", "MICCAI'24", "mono (DA)", "36.165 / 37.014", "0.959 / 0.960", "0.039 / 0.041", "", "100", "", "", "RTX 4090", "Endo-G2T Tab. 1", ""),
    (C, "Endo-G2T", "arXiv'25", "", "40.080 / 38.290", "0.982 / 0.970", "0.007 / 0.016", "", "148", "", "", "RTX 4090", "Endo-G2T Tab. 1", ""),
    (C, "EndoGS", "arXiv'24", "", "37.16 / 36.19", "0.953 / 0.941", "0.045 / 0.041", "", "", "", "", "RTX 4090", "EndoWave Tab. I", ""),
    (C, "SurgicalGS", "arXiv'24", "", "38.31 / 38.05", "0.962 / 0.959", "0.062 / 0.062", "", "183", "", "", "RTX 4090", "EndoWave Tab. I", ""),
    (C, "EndoWave", "arXiv'25", "", "38.93 / 38.51", "0.981 / 0.969", "0.010 / 0.021", "", "86", "", "", "RTX 4090", "EndoWave Tab. I", ""),
    (C, "HFGS", "arXiv'24", "", "37.83 / 38.44", "0.969 / 0.968", "0.022 / 0.043", "", "", "", "", "", "HFGS Tab. 2", ""),
    (C, "This work", "--", "", "", "", "", "", "", "", "", "", "--", "EndoNeRF data not available (request form)"),
]

HEAD = ["group", "method", "venue", "depth input", "PSNR", "SSIM", "LPIPS", "AbsRel", "FPS", "train time",
        "GPU mem", "GPU", "source", "note"]
with open("sota_comparison.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(HEAD); w.writerows(rows)

def tex(s):
    return (s.replace("±", r"$\pm$").replace("~", r"$\sim$").replace("_", r"\_").replace("%", r"\%")
             .replace("&", r"\&").replace("*", r"$^{*}$").replace("'", "'"))

SECTION = {
    A: r"\textbf{A. StereoMIS P1, frames $\sim$800--1000 (same clip as this work).} Different papers ran different baselines on this clip; rows are grouped by source.",
    B: r"\textbf{B. StereoMIS, other sequences and protocols} (not comparable with Part A or with each other; clip shown per row or in the note).",
    C: r"\textbf{C. EndoNeRF dataset} (cutting\,/\,pulling).",
}

out = [r"""\documentclass[9pt]{extarticle}
\usepackage[a4paper,landscape,margin=1.3cm]{geometry}
\usepackage{booktabs,longtable,array}
\usepackage[table]{xcolor}
\usepackage{times}
\definecolor{ours}{HTML}{2A78D6}
\definecolor{grp}{HTML}{EEF1F5}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.08}
\pagestyle{plain}
\begin{document}
\begin{center}{\large\textbf{Endoscopic dynamic scene reconstruction: comparison with the state of the art}}\\[2pt]
\small Third-party numbers are copied from each cited paper (arXiv HTML); a blank cell means not reported. PSNR/SSIM/LPIPS on held-out frames; SSIM printed in \% by some papers is converted to 0--1.\\
AbsRel: ours is the rendered depth vs.\ SGBM stereo depth after per-frame median scaling; SurgicalGS's is vs.\ RAFT stereo depth (different definition). FPS, time and memory are on each paper's own GPU.\end{center}
\scriptsize
\begin{longtable}{@{}p{4.2cm}p{1.15cm}p{2.1cm}p{2.2cm}p{2.0cm}p{2.0cm}p{1.35cm}p{0.85cm}p{1.15cm}p{1.0cm}p{1.35cm}p{2.3cm}p{3.6cm}@{}}
\toprule
Method & Venue & Depth input & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & AbsRel$\downarrow$ & FPS & Train & Mem. & GPU & Source & Note \\
\midrule
\endhead
\bottomrule
\endfoot
"""]
last = None
for r in rows:
    g = r[0]
    if g != last:
        if last is not None:
            out.append(r"\midrule" + "\n")
        out.append(r"\multicolumn{13}{@{}l}{\cellcolor{grp}" + SECTION[g] + r"} \\" + "\n")
        last = g; prev_src = None
    src = r[12]
    if g == A and prev_src is not None and src != prev_src:
        out.append(r"\cmidrule(l{0pt}r{0pt}){1-13}" + "\n")
    prev_src = src
    cells = [tex(c) for c in r[1:]]
    ours = r[1].startswith("Ours")
    if ours:
        cells = [r"\textbf{" + c + "}" for c in cells]
        out.append(r"\rowcolor{ours!12}")
    out.append(" & ".join(cells) + r" \\" + "\n")
out.append(r"""\end{longtable}
\vspace{-4pt}
\noindent\textbf{Reading Part A.} The same method varies by up to 0.7\,dB between papers on this clip (Endo-4DGS: 31.98 ours, 32.17 Endo-PairGS, 32.19 Endo-G2T, 32.69 original), and its LPIPS ranges from 0.066 to 0.192, so differences below $\sim$1\,dB across papers are not reliable. Within one protocol, our best model improves Endo-4DGS by +0.42\,dB PSNR and cuts depth error by 37\,\%; the strongest monocular methods on this clip (Endo-PairGS 33.56, Endo-G2T 33.58, Deform3DGS 33.35 in Endo-PairGS, EndoWave 33.26) are about 1.2\,dB higher in PSNR, but none report geometric accuracy. Endo-TSR (38.45) uses stereo depth and is not a monocular method.\\[2pt]
\noindent\textbf{Sources} (arXiv unless noted): Endo-4DGS 2401.16416 (v3/v4); Endo-PairGS IJCARS 2026, doi:10.1007/s11548-026-03707-y; Endo-TSR 2609.32399; Endo-G2T 2511.21367; EndoWave 2510.23087; Deform3DGS 2405.17835; ColorGS 2508.18696; Diff2DGS 2602.18314; SurgicalGaussian 2407.05023; SGS 2503.04079; SurgicalGS 2410.09292; EH-SurGS 2501.01101; Endo-4DGX 2506.23308; Endo-GSMT MICCAI 2025 (paper 1655); EndoSurf 2307.11307; HFGS 2405.17872.
\end{document}
""")
open("sota_comparison.tex", "w").write("".join(out))
subprocess.run(["pdflatex", "-interaction=nonstopmode", "sota_comparison.tex"], stdout=subprocess.DEVNULL)
subprocess.run(["pdflatex", "-interaction=nonstopmode", "sota_comparison.tex"], stdout=subprocess.DEVNULL)
print(len(rows), "rows")
