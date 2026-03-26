#!/usr/bin/env bash
set -euo pipefail

# Simple, course-friendly bootstrap.
# - creates .env from .env.example
# - creates venv (./.venv)
# - installs project in editable mode with recommended extras

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "[bootstrap] Created .env from .env.example"
  else
    echo "[bootstrap] ERROR: .env.example not found" >&2
    exit 1
  fi
fi

PY=${PYTHON:-python3}

# Default extras for the course.
# - agents: optional smolagents backend
# - g4f: open/free LLM access (can be unstable, but useful for the course)
EXTRAS="dev,agents,g4f"

# Optional flags:
#   --gnn      -> install PyTorch Geometric (GNN mode)
#   --gnn-ext  -> also install PyG extension libraries (pyg-lib, torch-scatter, ...)
#   --mm       -> multimodal PDF parsing (PyMuPDF)
#   --agents-hf -> install smolagents[transformers] for local HF models
for arg in "$@"; do
  case "$arg" in
    --gnn)
      EXTRAS="${EXTRAS},gnn"
      ;;
    --gnn-ext|--gnn_ext)
      EXTRAS="${EXTRAS},gnn,gnn_ext"
      ;;
    --mm)
      EXTRAS="${EXTRAS},mm"
      ;;
    --agents-hf|--agents_hf|--hf)
      EXTRAS="${EXTRAS},agents_hf"
      ;;
  esac
done

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  echo "[bootstrap] Created venv at .venv"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip wheel setuptools

# Install with sensible defaults.
python -m pip install -e ".[${EXTRAS}]"

python -c "import scireason; print('[bootstrap] Import OK')"

cat <<'MSG'

[bootstrap] Done.
Next steps:
  1) (optional) docker compose up -d   # for Neo4j/Qdrant/GROBID
  2) top-papers-graph demo-run         # fully offline demo pipeline

Optional extras:
  ./scripts/bootstrap.sh --gnn      # installs PyG (GNN mode)
  ./scripts/bootstrap.sh --gnn-ext  # installs PyG + extensions (best performance)
  ./scripts/bootstrap.sh --mm       # multimodal PDF parsing
  ./scripts/bootstrap.sh --agents-hf  # local HF models for smolagents
MSG
