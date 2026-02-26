#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-0}"

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required"
  exit 1
fi

if command -v nvpmodel >/dev/null 2>&1; then
  echo "Setting nvpmodel mode: ${MODE}"
  sudo nvpmodel -m "${MODE}"
else
  echo "nvpmodel not found; skipping"
fi

if command -v jetson_clocks >/dev/null 2>&1; then
  echo "Enabling jetson_clocks"
  sudo jetson_clocks
else
  echo "jetson_clocks not found; skipping"
fi

echo "Power setup complete."
