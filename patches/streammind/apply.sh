#!/usr/bin/env bash
set -euo pipefail
STREAMMIND_ROOT="${1:?Usage: $0 /path/to/StreamMind}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Applying StreamVAD patches to ${STREAMMIND_ROOT} ..."
cp -v "${SCRIPT_DIR}/streamvad_data.py" "${STREAMMIND_ROOT}/data/streamvad_data.py"
cd "${STREAMMIND_ROOT}"
git apply "${SCRIPT_DIR}/datasets.py.patch" && echo "datasets.py patched OK"
git apply "${SCRIPT_DIR}/train_new_stream.py.patch" && echo "train_new_stream.py patched OK"
echo "Done."
