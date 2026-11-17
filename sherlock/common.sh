# settings shared by every job, sourced by the sbatch scripts
# env: GLASSBOX_HOME (submit directory), CONTAINER, OVERRIDES (extra override yamls, relative to the checkout)

set -u

PROJECT_DIR="${GLASSBOX_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -f "${PROJECT_DIR}/scripts/train.py" ]; then
    echo "FATAL: ${PROJECT_DIR} is not a glassbox-ml checkout." >&2
    echo "Submit from the repository root, or export GLASSBOX_HOME." >&2
    exit 1
fi

CONTAINER="${CONTAINER:-/home/groups/jrosser/drone_dengue/segmentation_container_multiclass.sif}"
if [ ! -f "${CONTAINER}" ]; then
    echo "FATAL: container not found: ${CONTAINER}" >&2
    exit 1
fi

CONFIG_ARGS=(--config "${PROJECT_DIR}/configs/base_config.yaml" --override "${PROJECT_DIR}/configs/sherlock/sherlock.yaml")
for OVERRIDE in ${OVERRIDES:-}; do
    CONFIG_ARGS+=(--override "${PROJECT_DIR}/${OVERRIDE}")
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" # one BLAS thread per granted core
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

run_python() {
    singularity exec --env PYTHONPATH="${PROJECT_DIR}${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}" \
        "${CONTAINER}" python -u "$@"
}

echo "Job ${SLURM_JOB_ID:-<none>} on ${SLURM_NODELIST:-$(hostname)}, ${OMP_NUM_THREADS} threads, $(date)"
echo "Code:    ${PROJECT_DIR} $(cd "${PROJECT_DIR}" && git log --oneline -1 2>/dev/null || echo '(not a git checkout)')"
