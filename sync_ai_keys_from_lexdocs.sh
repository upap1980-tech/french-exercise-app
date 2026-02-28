#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_ENV="/Users/victormfrancisco/Desktop/PROYECTOS/LexDocsPro-LITE/.env"
DST_ENV="$ROOT_DIR/backend/.env"

if [[ ! -f "$SRC_ENV" ]]; then
  echo "No existe origen: $SRC_ENV" >&2
  exit 1
fi

if [[ ! -f "$DST_ENV" ]]; then
  echo "No existe destino: $DST_ENV" >&2
  exit 1
fi

TMP_FILE="$(mktemp)"
cp "$DST_ENV" "$TMP_FILE"

KEYS=(
  "PERPLEXITY_API_KEY"
  "OPENAI_API_KEY"
  "GEMINI_API_KEY"
  "DEEPSEEK_API_KEY"
  "ANTHROPIC_API_KEY"
  "GROQ_API_KEY"
  "GLM_API_KEY"
  "QWEN_API_KEY"
  "KIMI_API_KEY"
  "KLING_API_KEY"
  "HUGGINGFACE_API_KEY"
  "MANUS_API_KEY"
)

copied=0
for key in "${KEYS[@]}"; do
  value="$(awk -F= -v key="$key" '$1==key {sub(/^[^=]*=/, "", $0); print $0; exit}' "$SRC_ENV" || true)"
  if [[ -n "${value:-}" ]]; then
    if grep -q "^${key}=" "$TMP_FILE"; then
      sed -i '' "s|^${key}=.*|${key}=${value}|" "$TMP_FILE"
    else
      printf "%s=%s\n" "$key" "$value" >> "$TMP_FILE"
    fi
    copied=$((copied + 1))
    echo "Copiada: $key"
  fi
done

cp "$TMP_FILE" "$DST_ENV"
rm -f "$TMP_FILE"
echo "Sincronización completada. Variables copiadas: $copied"
