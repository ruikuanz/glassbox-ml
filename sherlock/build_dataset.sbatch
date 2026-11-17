#!/bin/bash
# build the tile dataset: every image/mask pair under paths.base_path -> tiles + bounding boxes
# env: OVERRIDES (extra override yamls)

#SBATCH --job-name=gb_build
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --account=jrosser

source "${GLASSBOX_HOME:-${SLURM_SUBMIT_DIR}}/sherlock/common.sh"

run_python "${PROJECT_DIR}/scripts/build_dataset.py" "${CONFIG_ARGS[@]}"
EXIT_CODE=$?
echo "End: $(date), exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
