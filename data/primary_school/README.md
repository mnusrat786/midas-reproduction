# Primary School Temporal Network

Download from: https://sociopatterns.org/datasets/primary-school-temporal-network-data/

Place these files here:
- `primaryschool.csv`          — 125,773 rows, tab-separated: t i j Ci Cj
- `primaryschool_metadata.txt` — 242 rows, tab-separated: node_id class gender

Format:
- t  = timestamp in seconds (20-second intervals)
- i,j = anonymous node IDs
- Ci,Cj = class labels (1A, 1B, ..., 5B, Teachers)
- gender = M / F / Unknown
