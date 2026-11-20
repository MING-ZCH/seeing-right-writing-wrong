#!/usr/bin/env bash
# Create the two isolated GPU venvs (.venv_main, .venv_qwen3vl) in the repository root (does not touch existing envs).
# Usage: bash scripts/setup_envs.sh [main|qwen3vl|all] ; WITH_VLLM=1 also installs the optional vLLM pin.
set -euo pipefail
cd "$(dirname "$0")/.."
WHICH="${1:-all}"
PYBIN="${PYBIN:-python3}"
mk() {  # mk <dir> <req> [<optional req>]
  local d="$1" req="$2" opt="${3:-}"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$d" --python "${UV_PYTHON:-3.10}"
    uv pip install --python "$d/bin/python" -r "$req"
    [[ -n "${WITH_VLLM:-}" && -n "$opt" ]] && uv pip install --python "$d/bin/python" -r "$opt"
  else
    "$PYBIN" -m venv "$d"
    "$d/bin/pip" install -U pip
    "$d/bin/pip" install -r "$req"
    [[ -n "${WITH_VLLM:-}" && -n "$opt" ]] && "$d/bin/pip" install -r "$opt"
  fi
  "$d/bin/python" -c "import torch, transformers; print('$d', torch.__version__, transformers.__version__, torch.cuda.is_available())"
}
[[ "$WHICH" == main || "$WHICH" == all ]] && mk .venv_main requirements/requirements_main.txt requirements/requirements_main_vllm.txt
[[ "$WHICH" == qwen3vl || "$WHICH" == all ]] && mk .venv_qwen3vl requirements/requirements_qwen3vl.txt requirements/requirements_qwen3vl_vllm.txt
