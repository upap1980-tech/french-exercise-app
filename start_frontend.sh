#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/setup_dev_env.sh" frontend

cd "$SCRIPT_DIR/frontend"
exec npm run dev
