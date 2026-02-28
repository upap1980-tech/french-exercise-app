#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_URL="http://localhost:5012/api/health"
FRONTEND_URL="http://localhost:5191"

"$SCRIPT_DIR/setup_dev_env.sh" all

kill_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti :"$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
}

kill_port 5012
kill_port 5191

# Start backend in background
"$SCRIPT_DIR/start_backend.sh" &
BACKEND_PID=$!

cleanup() {
  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$SCRIPT_DIR/start_frontend.sh" &
FRONTEND_PID=$!

# Wait backend and frontend health
for _ in {1..40}; do
  if curl -sf "$BACKEND_URL" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
for _ in {1..40}; do
  if curl -sf "$FRONTEND_URL" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

# Open browser when stack is ready
open "http://localhost:5191" || true

# Keep terminal attached to frontend process
wait "$FRONTEND_PID"
