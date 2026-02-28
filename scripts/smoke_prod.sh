#!/usr/bin/env bash
set -euo pipefail
BASE_URL="http://localhost:8765"

echo "Checking /api/health..."
if curl -sfS "$BASE_URL/api/health" -o /dev/null; then
  echo "OK: /api/health"
else
  echo "FAIL: /api/health"; exit 2
fi

echo "Checking /chat... (expect 200 or redirect)"
status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/chat")
if [[ "$status" =~ ^(200|301|302)$ ]]; then
  echo "OK: /chat returned $status"
else
  echo "FAIL: /chat returned $status"; exit 2
fi

echo "Checking /api/v1/chat/conversations..."
status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/v1/chat/conversations")
if [[ "$status" =~ ^(200|401|403)$ ]]; then
  echo "OK: /api/v1/chat/conversations returned $status"
else
  echo "FAIL: /api/v1/chat/conversations returned $status"; exit 2
fi

echo "Smoke tests passed."
