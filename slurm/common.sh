# Shared environment setup for the SLURM scripts in this folder (sourced, not run).
#   VENV_PATH   path to the Python environment (required)
#   MODULES     space-separated modules to load first, e.g. "GCC/14.3.0 CUDA/12.9.1" (optional)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
if [ -n "${MODULES:-}" ]; then
    module purge
    # shellcheck disable=SC2086
    module load ${MODULES}
fi
source "${VENV_PATH:?set VENV_PATH to your Python environment}/bin/activate"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-none}"
