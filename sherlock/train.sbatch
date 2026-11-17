#!/bin/bash
# train the detector on cpu; checkpoints every epoch, no resume
# env: OVERRIDES (extra override yamls, e.g. configs/experiments/wide.yaml)

#SBATCH --job-name=gb_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --account=jrosser

source "${GLASSBOX_HOME:-${SLURM_SUBMIT_DIR}}/sherlock/common.sh"

run_python "${PROJECT_DIR}/scripts/train.py" "${CONFIG_ARGS[@]}"
EXIT_CODE=$?
echo "End: $(date), exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
