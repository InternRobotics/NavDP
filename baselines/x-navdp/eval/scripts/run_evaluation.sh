#!/usr/bin/env bash
# Run X-NavDP evaluation with Isaac Lab.

set -euo pipefail

CONFIG_FILE=""
SCENE_INDEX=""
SERVER_PORT=19999
DEVICE="cuda:0"
NUM_EPISODES=""
MAX_STEPS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --scene_index)
            SCENE_INDEX="$2"
            shift 2
            ;;
        --server_port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: --config_file is required"
    echo "Usage: bash scripts/run_evaluation.sh --config_file eval/config/eval_pointgoal/quadruped_clutter_easy.yaml"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

cd "${REPO_ROOT}"

echo "Running X-NavDP evaluation..."
echo "  Config: $CONFIG_FILE"
echo "  Scene Index: ${SCENE_INDEX:-<from config>}"
echo "  Server Port: $SERVER_PORT"
echo "  Device: $DEVICE"
echo "  Num Episodes: ${NUM_EPISODES:-<all samples>}"

EVAL_ARGS=(
    --config_file "$CONFIG_FILE"
    --server_port "$SERVER_PORT"
    --device "$DEVICE"
)
if [[ -n "$SCENE_INDEX" ]]; then
    EVAL_ARGS+=(--scene_index "$SCENE_INDEX")
fi
if [[ -n "$NUM_EPISODES" ]]; then
    EVAL_ARGS+=(--num_episodes "$NUM_EPISODES")
fi
if [[ -n "$MAX_STEPS" ]]; then
    EVAL_ARGS+=(--max_steps "$MAX_STEPS")
fi

python -m eval.scripts.evaluate_pointgoal "${EVAL_ARGS[@]}"
