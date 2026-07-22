#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_cmd="${PYTHON:-python3}"
venv_path="${VENV_PATH:-${repo_root}/.venv}"

cd "${repo_root}"

if [[ ! -r /proc/version ]] || ! grep -qi microsoft /proc/version; then
  echo "warning: this installer is intended for NVIDIA CUDA under WSL 2" >&2
fi
if ! command -v "${python_cmd}" >/dev/null 2>&1; then
  echo "error: ${python_cmd} is not available" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi is unavailable inside WSL; update the Windows NVIDIA driver first" >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv,noheader

if ! "${python_cmd}" -m venv "${venv_path}"; then
  echo "error: Python venv support is unavailable; install python3-venv and retry" >&2
  exit 1
fi

"${venv_path}/bin/python" -m pip install --upgrade pip
"${venv_path}/bin/python" -m pip install --upgrade -e ".[dev,cuda12]"
"${venv_path}/bin/python" "${repo_root}/scripts/verify_cuda.py"

cat <<EOF

CUDA environment is ready.
Activate it with:
  source "${venv_path}/bin/activate"
For direct JAX and pytest commands in WSL:
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
EOF