#!/usr/bin/env bash
# Distributed training launcher (8 GPUs on one node by default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${WORLD_SIZE:=1}"
: "${RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${NPROC_PER_NODE:=4}"
: "${SCENE_DIR:=${REPO_ROOT}/data/scenes}"

if [[ -n "${ACADOS_SOURCE_DIR:-}" ]]; then
    export ACADOS_SOURCE_DIR
    export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${WORLD_SIZE}" \
    --node_rank="${RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --rdzv_conf='timeout=3600' \
    train.py \
    --scene_dir "${SCENE_DIR}" \
    "$@"
