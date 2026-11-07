#!/usr/bin/env bash
# Rebuild everything from the raw CSV files: processed data, figures, models, tests.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
if [ ! -f data/raw/olist_orders_dataset.csv ]; then
  echo "Put the 7 Olist CSV files in data/raw/ first (see README)"; exit 1
fi
cd src
"$PY" prepare.py      # audit + cleaning + order-level table      -> data/processed, results/tables
"$PY" eda.py          # 6 questions, figures 01-10                -> results/figures
"$PY" model.py        # baseline + 3 models, temporal split        -> results/
"$PY" make_docs.py    # feature dictionary                         -> docs/
cd ..
dot -Tpng docs/er_diagram.dot -o results/figures/00_er_diagram.png 2>/dev/null || echo "graphviz not installed: ER diagram not re-rendered"
"$PY" -m pytest -q tests
