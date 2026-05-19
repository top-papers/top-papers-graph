#!/usr/bin/env bash
# Side-by-side демонстрация генерации гипотез: две явно заданные ветки.
#
# Прогоняет один и тот же набор кейсов (data/demo/hypothesis_demo_cases.json) на двух ветках:
#   LEFT  — baseline (по умолчанию demo/baseline-origin-main)
#   RIGHT — feature (по умолчанию feature/quality-pipeline-v3)
# Результат — runs/hypothesis_demo/report/report.html и report.md.
#
# Безопасность:
#   - отказывается работать, если есть незакоммиченные изменения,
#   - запоминает исходную ветку и ВСЕГДА возвращается на неё (trap),
#   - не делает stash / reset / clean.
#
# Переопределить ветки можно через env:
#   LEFT_BRANCH=... RIGHT_BRANCH=... ./scripts/eval/run_main_vs_feature_demo.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

CASES="${CASES:-data/demo/hypothesis_demo_cases.json}"
OUT_ROOT="${OUT_ROOT:-runs/hypothesis_demo}"
LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b-instruct}"
LEFT_BRANCH="${LEFT_BRANCH:-demo/baseline-origin-main}"
RIGHT_BRANCH="${RIGHT_BRANCH:-feature/quality-pipeline-v3}"
if [ -z "${PY:-}" ]; then
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PY="$REPO_ROOT/.venv/bin/python"
  else
    PY="python3"
  fi
fi
echo "[info] python: $PY"
echo "[info] LEFT  (baseline): $LEFT_BRANCH"
echo "[info] RIGHT (feature):  $RIGHT_BRANCH"

if [ ! -f "$CASES" ]; then
  echo "Не найден файл кейсов: $CASES" >&2
  exit 1
fi

# Чистое рабочее дерево обязательно — иначе git checkout затрёт правки.
# Исключаем auto-генерируемые файлы setuptools (egg-info): они регенерируются
# при pip install -e . и в норме одинаковы на ветках, не мешают checkout'у.
DIRTY_PATHSPEC=('.' ':(exclude,glob)**/*.egg-info' ':(exclude,glob)**/*.egg-info/**')
if ! git diff --quiet -- "${DIRTY_PATHSPEC[@]}" \
   || ! git diff --cached --quiet -- "${DIRTY_PATHSPEC[@]}"; then
  echo "Есть незакоммиченные изменения. Закоммить или отложи их вручную и повтори." >&2
  git status --short -- "${DIRTY_PATHSPEC[@]}" >&2
  exit 1
fi

# Обе ветки должны существовать локально.
for br in "$LEFT_BRANCH" "$RIGHT_BRANCH"; do
  if ! git rev-parse --verify --quiet "refs/heads/$br" >/dev/null; then
    echo "Не найдена ветка $br. Создай её или передай LEFT_BRANCH/RIGHT_BRANCH через env." >&2
    exit 1
  fi
done

ORIGINAL_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$ORIGINAL_BRANCH" = "HEAD" ]; then
  echo "Сейчас detached HEAD — переключись на нужную ветку перед запуском." >&2
  exit 1
fi

restore_branch() {
  CURRENT="$(git rev-parse --abbrev-ref HEAD)"
  if [ "$CURRENT" != "$ORIGINAL_BRANCH" ]; then
    echo "[cleanup] возвращаюсь на $ORIGINAL_BRANCH"
    git checkout "$ORIGINAL_BRANCH"
  fi
}
trap restore_branch EXIT

run_side() {
  local label="$1"
  echo "==> Прогон: $label"
  git checkout "$label"
  "$PY" scripts/eval/run_hypothesis_demo.py \
    --cases "$CASES" \
    --label "$label" \
    --out-root "$OUT_ROOT" \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL"
}

run_side "$LEFT_BRANCH"
run_side "$RIGHT_BRANCH"

git checkout "$ORIGINAL_BRANCH"
trap - EXIT

LEFT_SLUG="${LEFT_BRANCH//\//__}"
RIGHT_SLUG="${RIGHT_BRANCH//\//__}"

echo "==> Рендер side-by-side отчёта"
"$PY" scripts/eval/render_hypothesis_ab.py \
  --left  "$OUT_ROOT/$LEFT_SLUG/manifest.json" \
  --right "$OUT_ROOT/$RIGHT_SLUG/manifest.json" \
  --out   "$OUT_ROOT/report"

echo
echo "Готово. Открыть: $OUT_ROOT/report/report.html"
