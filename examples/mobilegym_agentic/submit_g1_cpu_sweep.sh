#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Submit with an explicitly chosen account/reservation, for example:
#   sbatch --account=<account> --reservation=<reservation> --partition=debug \
#     --export=ALL,RELAX_REPO_DIR=<Relax-repo>,MOBILEGYM_ENV_URL=https://<gateway-host>:4180,\
#MOBILEGYM_REPO_DIR=<repo>,MOBILEGYM_PYTHON=<python>,G1_OUTPUT_ROOT=<fresh-dir> \
#     examples/mobilegym_agentic/submit_g1_cpu_sweep.sh
#
# No GPU is requested.  G1 establishes browser/env capacity before the GPU
# topology validation stages, and should run on an otherwise quiet node.

#SBATCH --job-name=mobilegym-g1-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/mobilegym_e2e/slurmlogs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/mobilegym_e2e/slurmlogs/%x-%j.err
#SBATCH --no-requeue

set -euo pipefail

if [ -z "${RELAX_REPO_DIR:-}" ] || [ -z "${MOBILEGYM_ENV_URL:-}" ] || [ -z "${MOBILEGYM_REPO_DIR:-}" ] || [ -z "${MOBILEGYM_PYTHON:-}" ] || [ -z "${G1_OUTPUT_ROOT:-}" ]; then
    echo "ERROR: set RELAX_REPO_DIR, MOBILEGYM_ENV_URL, MOBILEGYM_REPO_DIR, MOBILEGYM_PYTHON, and a fresh G1_OUTPUT_ROOT." >&2
    exit 2
fi
RUNNER="${RELAX_REPO_DIR}/examples/mobilegym_agentic/run_g1_cpu_sweep.sh"
if [ ! -f "${RUNNER}" ]; then
    echo "ERROR: G1 runner is missing: ${RUNNER}" >&2
    exit 2
fi
if ! curl -sk --max-time 10 -o /dev/null "${MOBILEGYM_ENV_URL}"; then
    echo "ERROR: MobileGym environment is unreachable: ${MOBILEGYM_ENV_URL}" >&2
    exit 2
fi

echo "G1 job=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
exec bash "${RUNNER}"
