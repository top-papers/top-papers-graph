#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
mkdir -p outputs reports data/derived/training

python_gpu_count() {
  python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
}

run_with_torchrun() {
  local script="$1"
  shift
  local nproc
  nproc="$(python_gpu_count)"
  if [ -z "$nproc" ] || [ "$nproc" -lt 1 ]; then
    nproc=1
  fi
  torchrun --standalone --nproc_per_node="$nproc" "$script" "$@"
}
