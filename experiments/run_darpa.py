"""
Reproduce paper Figures 2 and 3 on the DARPA dataset.

Figures produced (saved to ../figures/):
  figure2_roc.png            — ROC curve: MIDAS-R, MIDAS, SedanSpot
  figure3_auc_ap_vs_time.png — AUC and Avg Precision vs running time

Expected runtime: ~30-40 minutes (Python).
Paper values (C++): MIDAS=0.13s, MIDAS-R=0.39s, SedanSpot=83.7s

Usage:
    python experiments/run_darpa.py
"""

import sys, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import MIDAS, MIDAS_R, run_sedanspot
from plots.plot_figures import plot_roc_curve, plot_auc_vs_time

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / 'data/darpa/darpa_processed.csv'
LABEL_PATH= ROOT / 'data/darpa/darpa_ground_truth.csv'
FIG_DIR   = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def load_darpa():
    print("Loading DARPA dataset...")
    t0   = time.time()
    data = pd.read_csv(DATA_PATH, header=None, names=['src','dst','ts'],
                       dtype={'src':np.int32,'dst':np.int32,'ts':np.int32})
    labels = pd.read_csv(LABEL_PATH, header=None, names=['label'],
                         dtype={'label':np.int32})
    print(f"  {len(data):,} edges loaded in {time.time()-t0:.1f}s")
    return data, labels['label'].values


def run_detector(detector, src, dst, ts):
    n = len(src)
    scores = np.zeros(n, dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n):
        scores[i] = detector(int(src[i]), int(dst[i]), int(ts[i]))
    return scores, time.perf_counter() - t0


if __name__ == '__main__':
    data, labels = load_darpa()
    src, dst, ts = data.src.values, data.dst.values, data.ts.values

    print("\nRunning MIDAS...")
    sc_m, t_m = run_detector(MIDAS(num_rows=2, num_cols=2719, seed=42), src, dst, ts)
    print(f"  AUC={roc_auc_score(labels, sc_m):.4f}  Time={t_m:.1f}s")

    print("\nRunning MIDAS-R...")
    sc_mr, t_mr = run_detector(MIDAS_R(num_rows=2, num_cols=2719, alpha=0.5, seed=42), src, dst, ts)
    print(f"  AUC={roc_auc_score(labels, sc_mr):.4f}  Time={t_mr:.1f}s")

    print("\nRunning SedanSpot (~15 min)...")
    sc_s, t_s = run_sedanspot(src, dst, ts, sample_size=500, num_walks=50, seed=42)
    print(f"  AUC={roc_auc_score(labels, sc_s):.4f}  Time={t_s:.1f}s")

    plot_roc_curve(labels, sc_m, sc_mr, sc_s,
                   save_path=FIG_DIR / 'figure2_roc.png')

    plot_auc_vs_time(labels, sc_m, t_m, sc_mr, t_mr, sc_s, t_s,
                     save_path=FIG_DIR / 'figure3_auc_ap_vs_time.png')

    print("\nDone! Figures saved to figures/")
