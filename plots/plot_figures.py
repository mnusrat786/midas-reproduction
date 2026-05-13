"""
Shared plotting functions for all paper figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
from pathlib import Path


def plot_roc_curve(labels, scores_midas, scores_midasr, scores_sedan,
                   save_path: Path = None):
    """Figure 2: ROC curve for MIDAS-R, MIDAS, SedanSpot."""
    fpr_m,  tpr_m,  _ = roc_curve(labels, scores_midas)
    fpr_mr, tpr_mr, _ = roc_curve(labels, scores_midasr)
    fpr_s,  tpr_s,  _ = roc_curve(labels, scores_sedan)

    auc_m  = auc(fpr_m,  tpr_m)
    auc_mr = auc(fpr_mr, tpr_mr)
    auc_s  = auc(fpr_s,  tpr_s)

    print(f"  MIDAS     AUC={auc_m:.4f}  (paper: 0.91)")
    print(f"  MIDAS-R   AUC={auc_mr:.4f}  (paper: 0.95)")
    print(f"  SedanSpot AUC={auc_s:.4f}  (paper: 0.64)")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_mr, tpr_mr, color='#e377c2', lw=2, label=f'Midas-R  (AUC={auc_mr:.2f})')
    ax.plot(fpr_m,  tpr_m,  color='#ff7f0e', lw=2, label=f'Midas    (AUC={auc_m:.2f})')
    ax.plot(fpr_s,  tpr_s,  color='#2ca02c', lw=2, label=f'SedanSpot (AUC={auc_s:.2f})')
    ax.plot([0,1],  [0,1],  'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Figure 2: ROC for DARPA dataset', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {Path(save_path).name}")
    plt.close()


def plot_auc_vs_time(labels, sc_m, t_m, sc_mr, t_mr, sc_s, t_s,
                     save_path: Path = None):
    """Figure 3: AUC and Average Precision vs running time."""
    auc_m  = roc_auc_score(labels, sc_m)
    auc_mr = roc_auc_score(labels, sc_mr)
    auc_s  = roc_auc_score(labels, sc_s)
    ap_m   = average_precision_score(labels, sc_m)
    ap_mr  = average_precision_score(labels, sc_mr)
    ap_s   = average_precision_score(labels, sc_s)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    for ax, vm, vmr, vs, ylabel, title in [
        (ax1, auc_m,  auc_mr,  auc_s,  'AUC',
         'Figure 3 (top): Accuracy (AUC) vs Time'),
        (ax2, ap_m,   ap_mr,   ap_s,   'Average Precision Score',
         'Figure 3 (bottom): Avg Precision vs Time'),
    ]:
        ax.scatter([t_m],  [vm],  s=120, color='#ff7f0e', zorder=5,
                   label=f'MIDAS ({vm:.2f}, {t_m:.1f}s)')
        ax.scatter([t_mr], [vmr], s=120, color='#e377c2', zorder=5,
                   label=f'MIDAS-R ({vmr:.2f}, {t_mr:.1f}s)')
        ax.scatter([t_s],  [vs],  s=120, color='#2ca02c', marker='^', zorder=5,
                   label=f'SedanSpot ({vs:.2f}, {t_s:.1f}s)')
        ax.set_xscale('log')
        ax.set_xlabel('Running Time (seconds, log scale)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {Path(save_path).name}")
    plt.close()


def plot_scalability(nums, times_midas, times_midasr, save_path: Path = None):
    """Figure 4: Scalability — log-log line plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(nums, times_midas,  'o-', color='#CC0066', lw=2, ms=6, label='Midas')
    ax.plot(nums, times_midasr, 's-', color='#FFA500', lw=2, ms=6, label='Midas-R')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    xticks = [2**p for p in range(12, 23)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'$2^{{{p}}}$' for p in range(12, 23)], fontsize=8)
    ax.set_xlabel('Number of Edges', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Figure 4: Midas and Midas-R scale linearly\nwith the number of edges',
                 fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {Path(save_path).name}")
    plt.close()


def plot_processing_times(save_path: Path = None):
    """Figure 5: Grouped bar chart of per-edge processing times."""
    total = 4.554344
    midas_1, midas_2     = 4.4, 0.15
    midasr_1, midasr_2   = 4.3, 0.23
    midas_gt  = total - midas_1  - midas_2
    midasr_gt = total - midasr_1 - midasr_2

    x, width = np.arange(3), 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width/2, [midasr_1, midasr_2, midasr_gt], width,
           color='#FFA500', label='Midas-R')
    ax.bar(x + width/2, [midas_1,  midas_2,  midas_gt],  width,
           color='#CC0066', label='Midas')
    ax.set_xlabel('Time (microseconds)', fontsize=12)
    ax.set_ylabel('Frequency (millions of edges)', fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(['1', '2', '>2'], fontsize=12)
    ax.set_ylim(0, 5); ax.set_yticks([0,1,2,3,4,5])
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.title('Figure 5: Distribution of processing times for $\\sim$ 4.5$M$\n'
              'edges of DARPA dataset.', fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {Path(save_path).name}")
    plt.close()
