#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT_DIR/config/skillforge.env}"
CHECK_ONLY=0

if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
  CONFIG_FILE="${2:-$ROOT_DIR/config/skillforge.env}"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing configuration: $CONFIG_FILE" >&2
  echo "Copy config/skillforge.env.example to config/skillforge.env first." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${PYTHON_BIN:=python3}"
: "${VENV_DIR:=.venv}"
: "${INSTALL_TRAINING:=1}"
: "${INSTALL_LLM:=0}"
: "${INSTALL_GPU:=0}"

case "$INSTALL_TRAINING:$INSTALL_LLM:$INSTALL_GPU" in
  0:0:0|0:1:0|1:0:0|1:1:0|1:0:1|1:1:1) ;;
  *) echo "INSTALL_TRAINING, INSTALL_LLM, and INSTALL_GPU must be 0 or 1." >&2; exit 2 ;;
esac

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ "$CHECK_ONLY" == 1 ]]; then
  echo "Configuration is valid: $CONFIG_FILE"
  echo "Python: $PYTHON_BIN"
  echo "Virtual environment: $VENV_DIR"
  echo "Training extra: $INSTALL_TRAINING; LLM extra: $INSTALL_LLM; GPU extra: $INSTALL_GPU"
  exit 0
fi

cd "$ROOT_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$ROOT_DIR/$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -e .

if [[ "$INSTALL_TRAINING" == 1 ]]; then
  "$VENV_PYTHON" -m pip install -e '.[training]'
fi
if [[ "$INSTALL_LLM" == 1 ]]; then
  "$VENV_PYTHON" -m pip install -e '.[llm]'
fi
if [[ "$INSTALL_GPU" == 1 ]]; then
  "$VENV_PYTHON" -m pip install -e '.[gpu]'
fi

CUDA_VISIBLE_DEVICES="" "$VENV_PYTHON" -m unittest discover -s tests -v
echo
echo "SkillForge environment is ready."
echo "Activate it with: source $VENV_DIR/bin/activate"
echo "External model, dataset, benchmark-service, and API settings remain caller-supplied."
