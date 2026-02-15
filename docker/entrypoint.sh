#!/usr/bin/env bash
set -euo pipefail

# Simple dependency wait-loop so the stack works "out of the box" with docker compose.
# You can disable individual waits by setting WAIT_FOR_<SERVICE>=0.

WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-120}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-2}"

wait_for_http() {
  local name="$1"
  local url="$2"
  local timeout="$3"

  echo "[entrypoint] Waiting for ${name} at ${url} (timeout=${timeout}s)"
  local start
  start=$(date +%s)
  while true; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      echo "[entrypoint] ${name} is ready"
      return 0
    fi
    local now
    now=$(date +%s)
    if [ $((now - start)) -ge "$timeout" ]; then
      echo "[entrypoint] WARN: ${name} is not reachable yet (continuing anyway)"
      return 0
    fi
    sleep "$WAIT_INTERVAL_SECONDS"
  done
}

maybe_wait() {
  local flag="$1"; shift
  if [ "${flag}" = "0" ]; then
    return 0
  fi
  wait_for_http "$@"
}

# Qdrant
QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
maybe_wait "${WAIT_FOR_QDRANT:-1}" "Qdrant" "${QDRANT_URL%/}/readyz" "$WAIT_TIMEOUT_SECONDS"

# GROBID
GROBID_URL="${GROBID_URL:-http://grobid:8070}"
maybe_wait "${WAIT_FOR_GROBID:-1}" "GROBID" "${GROBID_URL%/}/api/isalive" "$WAIT_TIMEOUT_SECONDS"

# Neo4j (HTTP port is easier to probe than Bolt)
NEO4J_HTTP_URL="${NEO4J_HTTP_URL:-http://neo4j:7474}"
maybe_wait "${WAIT_FOR_NEO4J:-1}" "Neo4j" "${NEO4J_HTTP_URL%/}/" "$WAIT_TIMEOUT_SECONDS"

exec "$@"
