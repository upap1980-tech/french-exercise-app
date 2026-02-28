#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/setup_dev_env.sh" backend

cd "$SCRIPT_DIR/backend"

(venv/bin/python app.py > /tmp/french_backup_now.log 2>&1 &) 
BACK_PID=$!
trap 'kill "$BACK_PID" 2>/dev/null || true' EXIT INT TERM
sleep 2
curl -s -X POST http://localhost:5012/api/backups/export
