#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/setup_dev_env.sh" backend

cd "$SCRIPT_DIR/backend"

exec venv/bin/python app.py
