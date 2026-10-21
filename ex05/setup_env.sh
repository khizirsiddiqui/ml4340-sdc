#!/bin/bash
# Creates ./.venv with everything the pipeline needs (replaces the singularity image).
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.11 .venv
        uv pip install --python .venv/bin/python -r requirements.txt
    else
        python3 -m venv .venv
        .venv/bin/pip install -r requirements.txt
    fi
fi
