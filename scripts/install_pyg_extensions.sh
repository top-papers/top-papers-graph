#!/usr/bin/env bash
set -euo pipefail

# Install PyG extension wheels (pyg-lib, torch-scatter, torch-sparse, ...)
# using the official wheel index at https://data.pyg.org
#
# This script is optional. The repo works without it.

PY=${PYTHON:-python3}

TORCH_VER="$($PY -c "import torch; print(torch.__version__.split('+')[0])")"
CUDA_VER="$($PY -c "import torch; print(torch.version.cuda or 'cpu')")"

if [[ "$CUDA_VER" == "cpu" || "$CUDA_VER" == "None" ]]; then
  CUDA_TAG="cpu"
else
  CUDA_TAG="cu${CUDA_VER//./}"
fi

BASE_VER="$(echo "$TORCH_VER" | awk -F. '{print $1"."$2".0"}')"

URL1="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
URL2="https://data.pyg.org/whl/torch-${BASE_VER}+${CUDA_TAG}.html"

python -m pip install -U pip

set +e
OK=0
for URL in "$URL1" "$URL2"; do
  echo "[install_pyg_extensions] Trying: $URL"
  python -m pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "$URL"
  if [[ $? -eq 0 ]]; then
    OK=1
    break
  fi
  echo "[install_pyg_extensions] Failed for $URL, trying fallback..."
  echo
  sleep 1
done
set -e

if [[ $OK -ne 1 ]]; then
  echo "[install_pyg_extensions] ERROR: Could not install PyG extension wheels." >&2
  echo "See docs/gnn.md and PyG installation docs for manual instructions." >&2
  exit 1
fi

# Ensure torch_geometric itself is installed.
python -m pip install -U torch_geometric

echo "[install_pyg_extensions] Done."
