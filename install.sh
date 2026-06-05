#!/usr/bin/env bash
set -euo pipefail

REQUIRED_PYTHON="3.10"
VENV_DIR=".venv"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

find_python() {
  for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
      then
        printf '%s\n' "${candidate}"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON_BIN="$(find_python || true)"
if [ -z "${PYTHON_BIN}" ]; then
  echo "Error: Python ${REQUIRED_PYTHON}+ is required, but no compatible python executable was found." >&2
  echo "Install Python ${REQUIRED_PYTHON}+ and rerun ./install.sh." >&2
  exit 1
fi

echo "Using Python: $("${PYTHON_BIN}" --version)"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install -r requirements.txt

SITE_PACKAGES="$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
printf '%s\n' "$(pwd)" > "${SITE_PACKAGES}/ppocrv6_dxnn_repo.pth"

echo
echo "Environment ready."
echo "Activate it with: source ${VENV_DIR}/bin/activate"
echo "For DXNN commands, export DX_RT_PATH=/path/to/dx-all-suite/dx-runtime/dx_rt"
