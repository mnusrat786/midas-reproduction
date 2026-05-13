# DARPA Dataset

Download from: https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset

Place these files here:
- `darpa_processed.csv`    — 4,554,344 rows, columns: src, dst, timestamp (no header)
- `darpa_ground_truth.csv` — 4,554,344 rows, column: label (0=normal, 1=attack)
- `darpa_shape.txt`        — single integer: 4554344

To generate from the original zip, run:
```bash
python MIDAS/util/PreprocessData.py
```
