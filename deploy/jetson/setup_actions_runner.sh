#!/usr/bin/env bash
set -euo pipefail

# Registers a GitHub Actions self-hosted runner on Jetson (arm64).
#
# Required env:
#   RUNNER_TOKEN   Registration token from:
#                  GitHub repo -> Settings -> Actions -> Runners -> New self-hosted runner
#
# Optional env:
#   REPO_URL       Default: https://github.com/JafarBanar/mini-ai-deploy-pipeline
#   RUNNER_DIR     Default: $HOME/actions-runner
#   RUNNER_NAME    Default: $(hostname)-jetson
#   RUNNER_LABELS  Default: self-hosted,linux,arm64,jetson

REPO_URL="${REPO_URL:-https://github.com/JafarBanar/mini-ai-deploy-pipeline}"
RUNNER_DIR="${RUNNER_DIR:-$HOME/actions-runner}"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)-jetson}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,linux,arm64,jetson}"

if [ -z "${RUNNER_TOKEN:-}" ]; then
  echo "RUNNER_TOKEN is required."
  exit 1
fi

mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

if [ ! -f "./config.sh" ]; then
  echo "Downloading latest actions runner release metadata..."
  DL_URL="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | grep browser_download_url | grep 'linux-arm64' | head -n1 | cut -d '\"' -f 4)"
  if [ -z "$DL_URL" ]; then
    echo "Could not resolve linux-arm64 runner download URL."
    exit 1
  fi
  echo "Downloading: $DL_URL"
  curl -fsSL -o actions-runner.tar.gz "$DL_URL"
  tar xzf actions-runner.tar.gz
fi

echo "Configuring runner..."
./config.sh \
  --unattended \
  --replace \
  --url "$REPO_URL" \
  --token "$RUNNER_TOKEN" \
  --name "$RUNNER_NAME" \
  --labels "$RUNNER_LABELS"

echo "Installing and starting service..."
sudo ./svc.sh install
sudo ./svc.sh start

echo "Runner setup complete."
echo "Check status: sudo ./svc.sh status"

