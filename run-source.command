#!/bin/bash
# Run LUMEN RAW from source on an Apple silicon Mac (Python 3.12). First run creates .venv-macos,
# installs requirements-macos.txt and restores the runtime assets and the Perl ExifTool.
cd "$(dirname "$0")" || exit 1
PYTHON_BASE="${LUMEN_PYTHON:-python3.12}"
if [ ! -x .venv-macos/bin/python ]; then
    command -v "$PYTHON_BASE" >/dev/null || { echo "Install Python 3.12 (python.org, Homebrew or uv) and run again."; exit 1; }
    "$PYTHON_BASE" -m venv .venv-macos || exit 1
fi
PY=.venv-macos/bin/python
if ! "$PY" -c "import numpy, cv2, PIL, rawpy, PySide6.QtMultimedia, tifffile, onnxruntime, Metal" >/dev/null 2>&1; then
    "$PY" -m pip install --disable-pip-version-check --no-cache-dir -r requirements-macos.txt || exit 1
fi
"$PY" tools/fetch_assets.py --verify-only >/dev/null 2>&1 || "$PY" tools/fetch_assets.py || exit 1
"$PY" tools/fetch_exiftool.py || exit 1
exec "$PY" main.py "$@"
