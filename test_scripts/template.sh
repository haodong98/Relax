#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --time=00:59:00
#SBATCH --job-name=test
#SBATCH --output=/iopsstor/scratch/cscs/%u/rltest/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/rltest/%x-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

echo "START TIME: $(date)"
echo "JOBID=$SLURM_JOB_ID"
echo "NODELIST=$SLURM_NODELIST"
echo "NNODES=$SLURM_JOB_NUM_NODES"

NCCL_TESTS_DIR=${NCCL_TESTS_DIR:-/user-environment/env/default/bin}
NCCL_TESTS_SUFFIX=${NCCL_TESTS_SUFFIX:-}
SLURM_RESERVATION=${SLURM_RESERVATION:-PA-2338-RL}
JOB_UENV=${JOB_UENV:-prgenv-gnu-openmpi/25.12:v1}
JOB_VIEW=${JOB_VIEW:-default}
SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-pmix}

# Communication plan this script targets:
#   TP -> AllReduce, 4 GPUs within one node
#   DP -> AllReduce, 8 GPUs across 8 nodes (1 GPU per node)
#   EP -> AllGather, 8 GPUs across 8 nodes (1 GPU per node)
#   PP -> Send/Recv, 2 GPUs across 2 nodes (1 GPU per node)
TP_SIZE=${TP_SIZE:-4}
DP_SIZE=${DP_SIZE:-8}
EP_SIZE=${EP_SIZE:-8}
PP_SIZE=${PP_SIZE:-2}

# Run modes:
#   baseline      -> minimal site-compatible environment
#   training_like -> current Slingshot/CXI tuning
#   both          -> run both modes sequentially
TEST_MODE=${TEST_MODE:-baseline}

# Optional supplement for the current Megatron config.
# Main flow already covers EP ReduceScatter because the allgather
# dispatcher path is effectively AG + RS. This flag is only for
# extra RS paths you may still want to benchmark separately.
INCLUDE_EXTRA_RS=${INCLUDE_EXTRA_RS:-0}

TP_ARGS=${TP_ARGS:-"-b 8M -e 256M -f 2 -g ${TP_SIZE} -w 10 -n 50 -a 2"}
DP_ARGS=${DP_ARGS:-"-b 8M -e 2G -f 2 -g 1 -w 10 -n 50 -a 2"}
EP_ARGS=${EP_ARGS:-"-b 8M -e 2G -f 2 -g 1 -w 10 -n 50 -a 2"}
PP_ARGS=${PP_ARGS:-"-b 8M -e 256M -f 2 -g 1 -w 10 -n 50 -a 2"}

DEBUG_DIR=/iopsstor/scratch/cscs/$USER/nccl-tests/$SLURM_JOB_ID
mkdir -p "$DEBUG_DIR"
cp "$0" "$DEBUG_DIR/"

# This script defaults to the OpenMPI uenv because it has been validated
# on Clariden to form a real multi-rank world with `srun --mpi=pmix`.
# The currently validated nccl-tests build uses the default binary names
# without a suffix. If you later switch to a `_mpi` build, override with:
#   --export=ALL,NCCL_TESTS_SUFFIX=_mpi
# If you later switch back to a Cray MPICH workflow, override with:
#   --export=ALL,JOB_UENV=prgenv-gnu/25.11:v1,SLURM_MPI_TYPE=pmi2
SRUN_COMMON=(--mpi="${SLURM_MPI_TYPE}" --cpu-bind=none --distribution=block:block --network=disable_rdzv_get)
if [[ -n "${SLURM_RESERVATION}" ]]; then
  SRUN_COMMON+=(--reservation="${SLURM_RESERVATION}")
fi
if [[ -n "${JOB_UENV}" ]]; then
  SRUN_COMMON+=(--uenv="${JOB_UENV}")
fi
if [[ -n "${JOB_VIEW}" ]]; then
  SRUN_COMMON+=(--view="${JOB_VIEW}")
fi

max_required_nodes=$DP_SIZE
if (( EP_SIZE > max_required_nodes )); then
  max_required_nodes=$EP_SIZE
fi
if (( PP_SIZE > max_required_nodes )); then
  max_required_nodes=$PP_SIZE
fi

if (( SLURM_JOB_NUM_NODES < max_required_nodes )); then
  echo "ERROR: allocation has ${SLURM_JOB_NUM_NODES} nodes, but this script needs at least ${max_required_nodes} nodes." >&2
  echo "Adjust #SBATCH --nodes or override DP_SIZE/EP_SIZE/PP_SIZE to fit the allocation." >&2
  exit 1
fi

configure_mode() {
  local mode="$1"

  if [[ "${JOB_UENV}" == *openmpi* ]]; then
    export PMIX_MCA_psec=${PMIX_MCA_psec:-native}
  else
    export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-0}
  fi

  case "$mode" in
    baseline)
      export NCCL_DEBUG=${NCCL_DEBUG_BASELINE:-WARN}
      export NCCL_NET=${NCCL_NET_BASELINE:-AWS Libfabric}
      unset NCCL_CROSS_NIC || true
      unset NCCL_PROTO || true
      unset NCCL_NET_GDR_LEVEL || true
      unset FI_CXI_DEFAULT_CQ_SIZE || true
      unset FI_CXI_DEFAULT_TX_SIZE || true
      unset FI_CXI_DISABLE_HOST_REGISTER || true
      unset FI_CXI_RX_MATCH_MODE || true
      unset FI_MR_CACHE_MONITOR || true
      ;;
    training_like)
      export NCCL_DEBUG=${NCCL_DEBUG_TRAINING:-INFO}
      export NCCL_NET=${NCCL_NET_TRAINING:-AWS Libfabric}
      export NCCL_CROSS_NIC=${NCCL_CROSS_NIC_TRAINING:-1}
      export NCCL_PROTO=${NCCL_PROTO_TRAINING:-^LL128}
      export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL_TRAINING:-SYS}
      export FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE_TRAINING:-131072}
      export FI_CXI_DEFAULT_TX_SIZE=${FI_CXI_DEFAULT_TX_SIZE_TRAINING:-16384}
      export FI_CXI_DISABLE_HOST_REGISTER=${FI_CXI_DISABLE_HOST_REGISTER_TRAINING:-1}
      export FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE_TRAINING:-software}
      export FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR_TRAINING:-userfaultfd}
      ;;
    *)
      echo "ERROR: unsupported TEST_MODE '${mode}'. Use baseline, training_like, or both." >&2
      exit 1
      ;;
  esac
}

print_mode_env() {
  local mode="$1"
  echo "TEST_MODE=$mode"
  echo "JOB_UENV=${JOB_UENV:-<unset>}"
  echo "JOB_VIEW=${JOB_VIEW:-<unset>}"
  echo "SLURM_MPI_TYPE=${SLURM_MPI_TYPE:-<unset>}"
  echo "NCCL_TESTS_DIR=${NCCL_TESTS_DIR:-<unset>}"
  echo "NCCL_TESTS_SUFFIX=${NCCL_TESTS_SUFFIX:-<unset>}"
  echo "PMIX_MCA_psec=${PMIX_MCA_psec:-<unset>}"
  echo "NCCL_DEBUG=${NCCL_DEBUG:-<unset>}"
  echo "NCCL_NET=${NCCL_NET:-<unset>}"
  echo "NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-<unset>}"
  echo "NCCL_PROTO=${NCCL_PROTO:-<unset>}"
  echo "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-<unset>}"
  echo "FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE:-<unset>}"
  echo "FI_CXI_DEFAULT_TX_SIZE=${FI_CXI_DEFAULT_TX_SIZE:-<unset>}"
  echo "FI_CXI_DISABLE_HOST_REGISTER=${FI_CXI_DISABLE_HOST_REGISTER:-<unset>}"
  echo "FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE:-<unset>}"
  echo "FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-<unset>}"
  echo ""
}

run_nccl_test() {
  local outdir="$1"
  local label="$2"
  local test_name="$3"
  local nodes="$4"
  local ntasks="$5"
  local ntasks_per_node="$6"
  local gpus_per_task="$7"
  local gpu_bind="$8"
  local args="$9"

  mkdir -p "$outdir"
  local bin_path="${NCCL_TESTS_DIR}/${test_name}${NCCL_TESTS_SUFFIX}"

  echo "===== ${label}: ${test_name} ====="
  echo "nodes=${nodes} ntasks=${ntasks} ntasks_per_node=${ntasks_per_node}"
  echo "gpus_per_task=${gpus_per_task} gpu_bind=${gpu_bind}"
  echo "args=${args}"
  echo "resolved_bin=${bin_path}"
  srun "${SRUN_COMMON[@]}" \
       --nodes="${nodes}" \
       --ntasks="${ntasks}" \
       --ntasks-per-node="${ntasks_per_node}" \
       --gpus-per-task="${gpus_per_task}" \
       --gpu-bind="${gpu_bind}" \
       "${bin_path}" ${args} 2>&1 | tee "${outdir}/${test_name}.log"
}

run_one_mode() {
  local mode="$1"
  local mode_dir="${DEBUG_DIR}/${mode}"

  configure_mode "$mode"
  mkdir -p "$mode_dir"
  print_mode_env "$mode" | tee "${mode_dir}/env.txt"

  echo "############################################################"
  echo "# MODE: ${mode}"
  echo "############################################################"

  echo ""
  echo "############################################################"
  echo "# PHASE 1: TP — intra-node AllReduce (1 process x 4 GPUs on one node)"
  echo "############################################################"
  run_nccl_test "${mode_dir}/tp_intra_node" "TP" all_reduce_perf 1 1 1 "$TP_SIZE" none "$TP_ARGS"

  echo ""
  echo "############################################################"
  echo "# PHASE 2A: EP baseline — inter-node AllGather/ReduceScatter (2 nodes x 1 GPU)"
  echo "############################################################"
  run_nccl_test "${mode_dir}/ep_baseline_2ranks" "EP baseline" all_gather_perf 2 2 1 1 single:1 "$EP_ARGS"
  run_nccl_test "${mode_dir}/ep_baseline_2ranks" "EP baseline" reduce_scatter_perf 2 2 1 1 single:1 "$EP_ARGS"

  echo ""
  echo "############################################################"
  echo "# PHASE 2B: EP group-size-correct — inter-node AllGather/ReduceScatter (${EP_SIZE} nodes x 1 GPU)"
  echo "############################################################"
  run_nccl_test "${mode_dir}/ep_group_size" "EP group-size" all_gather_perf "$EP_SIZE" "$EP_SIZE" 1 1 single:1 "$EP_ARGS"
  run_nccl_test "${mode_dir}/ep_group_size" "EP group-size" reduce_scatter_perf "$EP_SIZE" "$EP_SIZE" 1 1 single:1 "$EP_ARGS"

  echo ""
  echo "############################################################"
  echo "# PHASE 3A: DP baseline — inter-node AllReduce (2 nodes x 1 GPU)"
  echo "############################################################"
  run_nccl_test "${mode_dir}/dp_baseline_2ranks" "DP baseline" all_reduce_perf 2 2 1 1 single:1 "$DP_ARGS"

  echo ""
  echo "############################################################"
  echo "# PHASE 3B: DP group-size-correct — inter-node AllReduce (${DP_SIZE} nodes x 1 GPU)"
  echo "############################################################"
  run_nccl_test "${mode_dir}/dp_group_size" "DP group-size" all_reduce_perf "$DP_SIZE" "$DP_SIZE" 1 1 single:1 "$DP_ARGS"

  echo ""
  echo "############################################################"
  echo "# PHASE 4: PP — inter-node Send/Recv (${PP_SIZE} nodes x 1 GPU)"
  echo "############################################################"
  run_nccl_test "${mode_dir}/pp_inter_node" "PP" sendrecv_perf "$PP_SIZE" "$PP_SIZE" 1 1 single:1 "$PP_ARGS"

  if [[ "${INCLUDE_EXTRA_RS}" == "1" ]]; then
    echo ""
    echo "############################################################"
    echo "# OPTIONAL: extra ReduceScatter supplements"
    echo "############################################################"
    run_nccl_test "${mode_dir}/dp_baseline_2ranks" "DP baseline supplement" reduce_scatter_perf 2 2 1 1 single:1 "$DP_ARGS"
    run_nccl_test "${mode_dir}/dp_group_size" "DP group-size supplement" reduce_scatter_perf "$DP_SIZE" "$DP_SIZE" 1 1 single:1 "$DP_ARGS"
  fi
}

case "$TEST_MODE" in
  baseline|training_like)
    run_one_mode "$TEST_MODE"
    ;;
  both)
    run_one_mode baseline
    run_one_mode training_like
    ;;
  *)
    echo "ERROR: TEST_MODE must be one of: baseline, training_like, both." >&2
    exit 1
    ;;
esac

echo ""
echo "############################################################"
echo "# DONE — Results saved to: $DEBUG_DIR"
echo "#"
echo "# Recommended first comparison:"
echo "#   TP  : <mode>/tp_intra_node/all_reduce_perf.log"
echo "#   EP  : <mode>/ep_group_size/all_gather_perf.log"
echo "#   DP  : <mode>/dp_group_size/all_reduce_perf.log"
echo "#   PP  : <mode>/pp_inter_node/sendrecv_perf.log"
echo "#"
echo "# If baseline looks healthy but training is still slow,"
echo "# compare against TEST_MODE=training_like for tuning effects."
echo "############################################################"
echo "END TIME: $(date)"
