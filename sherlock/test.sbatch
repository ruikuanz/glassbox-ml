#!/bin/bash
# run the test suite inside the container; installs pytest into PYDEPS on first use
# env: PYDEPS (/scratch/groups/jrosser/glassbox_ml/pydeps)

#SBATCH --job-name=gb_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --account=jrosser

source "${GLASSBOX_HOME:-${SLURM_SUBMIT_DIR}}/sherlock/common.sh"

PYDEPS="${PYDEPS:-/scratch/groups/jrosser/glassbox_ml/pydeps}"
EXTRA_PYTHONPATH="${PYDEPS}"
if ! run_python -c "import pytest" 2>/dev/null; then
    echo "Installing pytest into ${PYDEPS} ..."
    mkdir -p "${PYDEPS}"
    singularity exec "${CONTAINER}" python -m pip install --quiet --target "${PYDEPS}" pytest \
        || { echo "FATAL: could not install pytest" >&2; exit 1; }
fi

run_python -c "import numpy, rasterio, yaml; print('numpy', numpy.__version__, ' rasterio', rasterio.__version__, ' yaml', yaml.__version__)"
cd "${PROJECT_DIR}"
run_python -m pytest -p no:cacheprovider -v
EXIT_CODE=$?
echo "End: $(date), exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
