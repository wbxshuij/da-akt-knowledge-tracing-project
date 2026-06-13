#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m src.da_akt_project.train --config configs/test.yaml
