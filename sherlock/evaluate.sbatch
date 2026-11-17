#!/bin/bash
# score a trained run on dev and test at the best-f1 threshold on dev, and draw sample predictions
# env: OVERRIDES (the same override yamls the run was trained with)

#SBATCH --job-name=gb_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --account=jrosser

source "${GLASSBOX_HOME:-${SLURM_SUBMIT_DIR}}/sherlock/common.sh"

run_python "${PROJECT_DIR}/scripts/evaluate.py" "${CONFIG_ARGS[@]}"
EXIT_CODE=$?
echo "End: $(date), exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
