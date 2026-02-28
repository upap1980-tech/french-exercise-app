#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

MODE="${1:-all}" # all | backend | frontend

ensure_brew() {
  if command -v brew >/dev/null 2>&1; then
    return
  fi
  echo "[setup] Homebrew no encontrado. Instalando..."
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [ -x /usr/local/bin/brew ]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

ensure_python312() {
  if command -v python3.12 >/dev/null 2>&1; then
    return
  fi
  ensure_brew
  echo "[setup] Instalando python@3.12..."
  brew list python@3.12 >/dev/null 2>&1 || brew install python@3.12
}

ensure_backend_system_deps() {
  ensure_brew
  # Requerido por cairosvg/cairocffi para convertir SVG->PNG en exportación PDF.
  if ! brew list cairo >/dev/null 2>&1; then
    echo "[setup] Instalando cairo (SVG->PNG para PDFs)..."
    brew install cairo
  fi
  if ! brew list libffi >/dev/null 2>&1; then
    echo "[setup] Instalando libffi..."
    brew install libffi
  fi
}

resolve_python_bin() {
  if command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12"
    return
  fi
  if [ -x /opt/homebrew/bin/python3.12 ]; then
    echo "/opt/homebrew/bin/python3.12"
    return
  fi
  if [ -x /usr/local/bin/python3.12 ]; then
    echo "/usr/local/bin/python3.12"
    return
  fi
  echo "python3"
}

setup_backend() {
  ensure_python312
  ensure_backend_system_deps
  local pybin
  pybin="$(resolve_python_bin)"
  echo "[setup] Python backend: $pybin"
  cd "$BACKEND_DIR"
  if [ ! -x "venv/bin/python" ]; then
    echo "[setup] Creando entorno virtual backend..."
    "$pybin" -m venv venv
  fi
  echo "[setup] Verificando dependencias backend..."
  if ! venv/bin/python -c "import flask, flask_sqlalchemy, flask_cors, dotenv, reportlab, cairosvg" >/dev/null 2>&1; then
    echo "[setup] Dependencias backend incompletas, intentando instalar..."
    venv/bin/python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
    venv/bin/python -m pip install -r requirements.txt >/dev/null 2>&1 || true
    # Refuerzo explícito para pipeline de PDF con imagen.
    venv/bin/python -m pip install cairosvg reportlab >/dev/null 2>&1 || true
  fi
  if ! venv/bin/python -c "import flask, flask_sqlalchemy, flask_cors, dotenv, reportlab, cairosvg" >/dev/null 2>&1; then
    echo "[setup] ERROR: No se pudieron resolver dependencias backend (modo offline sin paquetes previos)." >&2
    exit 1
  fi
  if [ -f "requirements-optional-ai.txt" ]; then
    echo "[setup] Instalando stack IA opcional (best-effort)..."
    venv/bin/python -m pip install -r requirements-optional-ai.txt >/dev/null 2>&1 || true
  fi
}

setup_frontend() {
  if ! command -v npm >/dev/null 2>&1; then
    ensure_brew
    echo "[setup] Instalando node..."
    brew list node >/dev/null 2>&1 || brew install node
  fi
  cd "$FRONTEND_DIR"
  echo "[setup] Verificando dependencias frontend..."
  if [ ! -d "node_modules/vite" ]; then
    echo "[setup] Dependencias frontend incompletas, intentando instalar..."
    npm install >/dev/null 2>&1 || true
  fi
  if [ ! -d "node_modules/vite" ]; then
    echo "[setup] ERROR: No se pudieron resolver dependencias frontend (modo offline sin node_modules)." >&2
    exit 1
  fi
}

case "$MODE" in
  backend)
    setup_backend
    ;;
  frontend)
    setup_frontend
    ;;
  all)
    setup_backend
    setup_frontend
    ;;
  *)
    echo "Uso: $0 [all|backend|frontend]" >&2
    exit 1
    ;;
esac

echo "[setup] Entorno listo ($MODE)."
