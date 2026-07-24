#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STREAMVAD_ROOT:-${REPO_ROOT}}"

python3 -m venv "${VENV_DIR:-.venv-streamvad}"
source "${VENV_DIR:-.venv-streamvad}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r StreamMind/requirements.txt

echo "Server Python environment is ready."
echo "Install CUDA, FlashAttention, DeepSpeed, and model-specific packages according to your server policy before training."
