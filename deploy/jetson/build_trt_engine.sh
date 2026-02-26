#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${ONNX_PATH:-artifacts/model.onnx}"
ENGINE_PATH="${ENGINE_PATH:-artifacts/model.plan}"
PRECISION="${PRECISION:-fp16}" # fp16 or fp32
WORKSPACE_MB="${WORKSPACE_MB:-2048}"
SHAPES="${SHAPES:-input:1x3x32x32}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec is required. Install TensorRT on Jetson/Orin first."
  exit 1
fi

mkdir -p "$(dirname "$ENGINE_PATH")"

PRECISION_FLAG=""
if [ "$PRECISION" = "fp16" ]; then
  PRECISION_FLAG="--fp16"
elif [ "$PRECISION" = "fp32" ]; then
  PRECISION_FLAG=""
else
  echo "Unsupported PRECISION=$PRECISION (use fp16 or fp32)"
  exit 1
fi

echo "Building TensorRT engine:"
echo "  ONNX_PATH=$ONNX_PATH"
echo "  ENGINE_PATH=$ENGINE_PATH"
echo "  PRECISION=$PRECISION"
echo "  SHAPES=$SHAPES"

trtexec \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --minShapes="$SHAPES" \
  --optShapes="$SHAPES" \
  --maxShapes="$SHAPES" \
  $PRECISION_FLAG \
  --memPoolSize=workspace:"$WORKSPACE_MB"

echo "Saved engine: $ENGINE_PATH"

