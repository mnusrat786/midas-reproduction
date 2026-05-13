"""
Apply MIDAS to new datasets: Primary School and Hospital.

Figures produced (saved to ../figures/):
  ps_scores_over_time.png    — anomaly scores over 2 school days
  ps_roc.png                 — ROC curve (cross-class contacts as anomalies)
  ps_top_anomalies.png       — top 20 anomalous contacts with class + gender
  ps_score_by_class.png      — score distribution per class
  ps_score_by_gender.png     — score distribution by gender (from metadata)
  hospital_scores_over_time.png
  hospital_roc.png           — ROC curve (patient-patient contacts as anomalies)
  hospital_top_anomalies.png
  hospital_score_by_role.png — score distribution by role (PAT/MED/NUR/ADM)

Expected runtime: ~15 seconds total.

Usage:
    python experiments/run_new_datasets.py
"""

import sys, time, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import MIDAS, MIDAS_R

ROOT    = Path(__file__).parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

PS_DATA  = ROOT / 'data/primary_school/primaryschool.csv'
PS_META  = ROOT / 'data/primary_school/primaryschool_metadata.txt'
HOS_DATA = ROOT / 'data/hospital/detailed_list_of_contacts_Hospital.dat'


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_primary_school():
    df = pd.read_csv(PS_DATA, sep='\t', header=None,
                     names=['t','i','j','Ci','Cj'])
    df = df.sort_values('t').reset_index(drop=True)

    meta = pd.read_csv(PS_META, sep='\t', header=None,
                       names=['id','class','gender'])
    meta_dict = meta.set_index('id')[['class','gender']].to_dict('index')

    df['Gi'] = df['i'].map(lambda x: meta_dict.get(x, {}).get('gender', 'Unknown'))
    df['Gj'] = df['j'].map(lambda x: meta_dict.get(x, {}).get('gender', 'Unknown'))
    df['tick']  = ((df['t'] - df['t'].min()) // 20).astype(int) + 1
    df['label'] = (df['Ci'] != df['Cj']).astype(int)   # cross-class = anomaly

    print(f"Primary School: {len(df):,} edges | {df['tick'].nunique()} ticks | "
          f"{df['label'].mean()*100:.1f}% cross-class")
    return df


def load_hospital():
    df = pd.read_csv(HOS_DATA, sep='\t', header=None,
                     names=['t','i','j','Si','Sj'])
    df = df.sort_values('t').reset_index(drop=True)
    df['tick']  = ((df['t'] - df['t'].min()) // 20).astype(int) + 1
    df['label'] = ((df['Si'] == 'PAT') & (df['Sj'] == 'PAT')).astype(int)

    print(f"Hospital: {len(df):,} edges | {df['tick'].nunique()} ticks | "
          f"{df['label'].mean()*100:.1f}% patient-patient")
    return df


# ── Runner ────────────────────────────────────────────────────────────────────

def run_both(df):
    src = df['i'].values.astype(int)
    dst = df['j'].values.astype(int)
    ts  = df['tick'].values.astype(int)
    n   = len(src)

    det_m  = MIDAS(num_rows=2, num_cols=1024, seed=42)
    det_mr = MIDAS_R(num_rows=2, num_cols=1024, alpha=0.5, seed=42)

    sc_m  = np.zeros(n, dtype=np.float32)
    sc_mr = np.zeros(n, dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n):
        sc_m[i] = det_m(int(src[i]), int(dst[i]), int(ts[i]))
    t_m = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n):
        sc_mr[i] = det_mr(int(src[i]), int(dst[i]), int(ts[i]))
    t_mr = time.perf_counter() - t0

    print(f"  MIDAS:   {t_m:.2f}s  AUC={roc_auc_score(df['label'], sc_m):.4f}")
    print(f"  MIDAS-R: {t_mr:.2f}s  AUC={roc_auc_score(df['label'], sc_mr):.4f}")
    return sc_m, sc_mr


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_scores_over_time(df, sc_m, sc_mr, title, fname):
    ticks    = df['tick'].values
    tick_vals = np.unique(ticks)
    hours    = tick_vals * 20 / 3600
    max_m    = np.array([sc_m[ticks==t].max()  for t in tick_vals])
    max_mr   = np.array([sc_mr[ticks==t].max() for t in tick_vals])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(hours, max_m,  color='#CC0066', lw=1.2, label='Midas')
    ax1.set_ylabel('Max Anomaly Score'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_yscale('symlog')
    ax2.plot(hours, max_mr, color='#FFA500', lw=1.2, label='Midas-R')
    ax2.set_xlabel('Time (hours)'); ax2.set_ylabel('Max Anomaly Score')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_yscale('symlog')
    plt.suptitle(f'{title} — Anomaly Scores over Time', fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname.name}")


def plot_roc(df, sc_m, sc_mr, title, fname):
    labels = df['label'].values
    fpr_m,  tpr_m,  _ = roc_curve(labels, sc_m)
    fpr_mr, tpr_mr, _ = roc_curve(labels, sc_mr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_mr, tpr_mr, color='#FFA500', lw=2,
            label=f'Midas-R (AUC={auc(fpr_mr,tpr_mr):.3f})')
    ax.plot(fpr_m,  tpr_m,  color='#CC0066', lw=2,
            label=f'Midas   (AUC={auc(fpr_m,tpr_m):.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'ROC — {title}'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname.name}")


def plot_top_anomalies(df, sc_mr, role_i, role_j, title, fname, top_n=20):
    df = df.copy(); df['score'] = sc_mr
    top = df.nlargest(top_n, 'score')
    has_gender = 'Gi' in df.columns
    if has_gender:
        labels = [f"{r['i']}({r[role_i]},{r['Gi']}) ↔ {r['j']}({r[role_j]},{r['Gj']})"
                  for _, r in top.iterrows()]
    else:
        labels = [f"{r['i']}({r[role_i]}) ↔ {r['j']}({r[role_j]})"
                  for _, r in top.iterrows()]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), top['score'].values, color='#CC0066', edgecolor='k', lw=0.3)
    ax.set_yticks(range(top_n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Anomaly Score (MIDAS-R)')
    ax.set_title(f'Top {top_n} Anomalous Contacts — {title}')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname.name}")


def plot_by_role(df, sc_mr, role_col, title, fname):
    df = df.copy(); df['score'] = sc_mr
    roles = sorted(df[role_col].unique())
    data  = [np.log1p(df[df[role_col]==r]['score'].values) for r in roles]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, tick_labels=roles, patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    colors = plt.cm.Set2(np.linspace(0, 1, len(roles)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xlabel('Role / Class'); ax.set_ylabel('log(1 + Anomaly Score)')
    ax.set_title(f'Score Distribution by Role — {title}')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("PRIMARY SCHOOL")
    print("=" * 55)
    ps = load_primary_school()
    sc_m_ps, sc_mr_ps = run_both(ps)
    plot_scores_over_time(ps, sc_m_ps, sc_mr_ps, 'Primary School',
                          FIG_DIR / 'ps_scores_over_time.png')
    plot_roc(ps, sc_m_ps, sc_mr_ps,
             'Primary School (cross-class = anomaly)',
             FIG_DIR / 'ps_roc.png')
    plot_top_anomalies(ps, sc_mr_ps, 'Ci', 'Cj', 'Primary School',
                       FIG_DIR / 'ps_top_anomalies.png')
    plot_by_role(ps, sc_mr_ps, 'Ci', 'Primary School',
                 FIG_DIR / 'ps_score_by_class.png')
    plot_by_role(ps, sc_mr_ps, 'Gi', 'Primary School (Gender)',
                 FIG_DIR / 'ps_score_by_gender.png')

    print("\n" + "=" * 55)
    print("HOSPITAL")
    print("=" * 55)
    hos = load_hospital()
    sc_m_h, sc_mr_h = run_both(hos)
    plot_scores_over_time(hos, sc_m_h, sc_mr_h, 'Hospital',
                          FIG_DIR / 'hospital_scores_over_time.png')
    plot_roc(hos, sc_m_h, sc_mr_h,
             'Hospital (patient-patient = anomaly)',
             FIG_DIR / 'hospital_roc.png')
    plot_top_anomalies(hos, sc_mr_h, 'Si', 'Sj', 'Hospital',
                       FIG_DIR / 'hospital_top_anomalies.png')
    plot_by_role(hos, sc_mr_h, 'Si', 'Hospital',
                 FIG_DIR / 'hospital_score_by_role.png')

    print("\nAll figures saved to figures/")
