#!/usr/bin/env bash
set -euo pipefail

echo "== Jetson environment check =="

if command -v nvpmodel >/dev/null 2>&1; then
  echo "nvpmodel: OK"
else
  echo "nvpmodel: MISSING"
fi

if command -v jetson_clocks >/dev/null 2>&1; then
  echo "jetson_clocks: OK"
else
  echo "jetson_clocks: MISSING"
fi

if command -v trtexec >/dev/null 2>&1; then
  echo "trtexec: OK"
else
  echo "trtexec: MISSING"
fi

python3 - << 'PY'
import importlib
mods = ["onnxruntime", "numpy", "torch"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: MISSING ({e})")
PY
