#!/usr/bin/env bash
set -euo pipefail

URL="https://agents4science.stanford.edu/assets/Agents4Science_Template.zip"
OUT="Agents4Science_Template.zip"
DEST="template"

mkdir -p "${DEST}"

echo "Downloading: ${URL}"
curl -L "${URL}" -o "${OUT}"

echo "Unzipping into: ${DEST}/"
unzip -o "${OUT}" -d "${DEST}"

echo "Done. Template files are in: paper/agents4science/${DEST}/"

