# MIDAS — Python Reimplementation & Reproduction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Paper](https://img.shields.io/badge/Paper-AAAI%202020-red.svg)](https://arxiv.org/abs/1911.04464)

Full Python reimplementation of **MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams** (AAAI 2020). All paper figures reproduced. Algorithm applied to two new real-world datasets.

> Bhatia, Hooi, Yoon, Shin, Faloutsos. *MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams.* AAAI 2020.

---

## Project Structure

```
midas-reproduction/
├── src/
│   ├── count_min_sketch.py   # Count-Min Sketch data structure
│   ├── midas.py              # MIDAS (NormalCore) — Algorithm 1
│   ├── midas_r.py            # MIDAS-R (RelationalCore) — Algorithm 2
│   └── sedanspot.py          # SedanSpot baseline
├── experiments/
│   ├── run_darpa.py          # Reproduce Figures 2 & 3 (DARPA)
│   └── run_new_datasets.py   # Apply MIDAS to Primary School & Hospital
├── plots/
│   └── plot_figures.py       # All shared plotting functions
├── data/
│   ├── darpa/                # Place DARPA files here (see data/darpa/README.md)
│   ├── primary_school/       # Place Primary School files here
│   ├── hospital/             # Place Hospital files here
│   └── twitter/              # Place TwitterSecurity file here
├── figures/                  # All output figures saved here
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/mnusrat786/midas-reproduction.git
cd midas-reproduction
pip install -r requirements.txt
```

---

## Datasets

### 1. DARPA (paper dataset)
- **Download:** [MIT Lincoln Laboratory](https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset)
- **Size:** 4,554,344 edges | 9,484 source IPs | 23,398 destination IPs
- **Format:** `src, dst, timestamp` (no header)
- **Labels:** 0 = normal, 1 = attack (60.1% anomalies)
- **Place in:** `data/darpa/`

### 2. Primary School Temporal Network (new)
- **Download:** [SocioPatterns](https://sociopatterns.org/datasets/primary-school-temporal-network-data/)
- **Size:** 125,773 edges | 242 nodes | 3,100 time ticks (20s intervals, 2 days)
- **Format:** `t i j Ci Cj` — tab-separated (t=seconds, Ci/Cj=class labels)
- **Metadata:** `node_id, class, gender` (primaryschool_metadata.txt)
- **Anomaly definition:** Cross-class contacts (children from different classes)
- **Place in:** `data/primary_school/`

### 3. Hospital Ward Contact Network (new)
- **Download:** [SocioPatterns](https://sociopatterns.org/datasets/hospital-ward-dynamic-contact-network/)
- **Size:** 32,424 edges | 75 nodes | 9,453 time ticks (20s intervals, 4 days)
- **Format:** `t i j Si Sj` — tab-separated (Si/Sj = PAT/MED/NUR/ADM)
- **Anomaly definition:** Patient-to-patient contacts (rare and bursty)
- **Place in:** `data/hospital/`

---

## Preprocessing

### DARPA
Already preprocessed. `darpa_processed.csv` has integer-encoded IPs and integer timestamps.
To re-preprocess from the original zip:
```bash
python MIDAS/util/PreprocessData.py
```

### Primary School
No preprocessing needed. `run_new_datasets.py` handles everything:
- Reads tab-separated file directly
- Converts seconds → 20-second tick index: `tick = (t - t_min) // 20 + 1`
- Joins with metadata to get class and gender per node
- Labels cross-class contacts as anomalies (`label = 1 if Ci != Cj`)

### Hospital
No preprocessing needed. `run_new_datasets.py` handles everything:
- Converts seconds → 20-second tick index
- Labels patient-to-patient contacts as anomalies (`label = 1 if Si == PAT and Sj == PAT`)

---

## How to Run

### Reproduce paper Figures 2 & 3 (DARPA)
```bash
python experiments/run_darpa.py
```
Expected runtime: ~30–40 min (Python). Paper used C++ (~0.5s for MIDAS).

### Generate Figures 1, 4, 5 (no long computation needed)
```bash
python -c "
import sys; sys.path.insert(0, '.')
from plots.plot_figures import plot_processing_times, plot_scalability
import numpy as np
from pathlib import Path

# Figure 5
plot_processing_times(save_path=Path('figures/figure5_per_edge_times.png'))

# Figure 4 (using pre-collected timing data)
nums = [2**p for p in range(12, 21)]
t_m  = [0.097, 0.202, 0.389, 0.876, 1.969, 3.399, 6.661, 14.898, 27.040]
t_mr = [0.189, 0.445, 0.792, 1.821, 3.860, 7.706, 14.714, 32.190, 56.491]
plot_scalability(nums, t_m, t_mr, save_path=Path('figures/figure4_scalability.png'))
"
```

### Apply MIDAS to new datasets (~15 seconds)
```bash
python experiments/run_new_datasets.py
```

---

## Results

### DARPA (Paper Reproduction)

| Method | AUC | Avg Precision | Time (Python) | Time (C++) |
|--------|-----|---------------|---------------|------------|
| MIDAS | 0.87 | 0.92 | ~100s | 0.13s |
| MIDAS-R | 0.96 | 0.98 | ~230s | 0.39s |
| SedanSpot | 0.60 | 0.78 | ~880s | 83.7s |

### New Datasets

| Dataset | Method | AUC | Runtime |
|---------|--------|-----|---------|
| Primary School | MIDAS | 0.65 | 2.8s |
| Primary School | MIDAS-R | 0.49 | 5.7s |
| Hospital | MIDAS | 0.75 | 0.74s |
| Hospital | MIDAS-R | 0.36 | 1.72s |

**Primary School:** MIDAS detects cross-class contact bursts (e.g. during lunch breaks). AUC=0.65 shows it captures these anomalous mixing events better than random.

**Hospital:** MIDAS detects patient-patient contact bursts (AUC=0.75). Patients normally only interact with nurses/doctors — sudden patient clusters are flagged correctly.

---

## Algorithm Overview

### Score Formula (Definition 1 in paper)

```
score(u, v, t) = (a_uv - s_uv/t)^2 * t^2 / (s_uv * (t-1))
```

| Symbol | Meaning |
|--------|---------|
| `a_uv` | edges (u→v) in **current** time tick |
| `s_uv` | edges (u→v) across **all** time ticks |
| `t` | current timestamp |

This is a chi-squared statistic testing whether the current tick rate matches the historical average.

### MIDAS vs MIDAS-R

| Feature | MIDAS | MIDAS-R |
|---------|-------|---------|
| Edge counts | ✅ | ✅ |
| Node counts (src + dst) | ❌ | ✅ |
| Temporal decay (α=0.5) | ❌ | ✅ |
| Memory | O(w·b) | O(3·w·b) |

### Count-Min Sketch Parameters
- `num_rows = 2` (hash functions)
- `num_cols = 2719` (buckets) → approximation error ν = 0.001
- `alpha = 0.5` (temporal decay for MIDAS-R)

---

## Citation

```bibtex
@inproceedings{bhatia2020midas,
  title={MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams},
  author={Siddharth Bhatia and Bryan Hooi and Minji Yoon and Kijung Shin and Christos Faloutsos},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2020}
}
```
