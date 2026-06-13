#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m src.da_akt_project.train --config configs/test.yaml
python -m src.da_akt_project.train --config configs/test_plain_akt_ablation.yaml
python scripts/compare_runs.py \
  --run_a outputs/test_run --name_a DA-AKT \
  --run_b outputs/test_plain_akt --name_b Plain-AKT \
  --output outputs/ablation_comparison.md
