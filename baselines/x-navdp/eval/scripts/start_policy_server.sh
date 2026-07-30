#!/usr/bin/env bash
# Start the X-NavDP policy server for evaluation.

set -euo pipefail

PORT=19999
EMBODIMENT="quadruped"
CHECKPOINT=""
VISUALIZATION=""
DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --embodiment)
            EMBODIMENT="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-visualization)
            VISUALIZATION="--no-visualization"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    echo "Usage: bash scripts/start_policy_server.sh --checkpoint /path/to/model.ckpt"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

cd "${REPO_ROOT}"

echo "Starting X-NavDP policy server..."
echo "  Port: $PORT"
echo "  Embodiment: $EMBODIMENT"
echo "  Checkpoint: $CHECKPOINT"
echo "  Device: $DEVICE"

SERVER_ARGS=(
    --port "$PORT"
    --embodiment "$EMBODIMENT"
    --checkpoint "$CHECKPOINT"
    --device "$DEVICE"
)
if [[ -n "$VISUALIZATION" ]]; then
    SERVER_ARGS+=("$VISUALIZATION")
fi

python -m eval.src.policy_server "${SERVER_ARGS[@]}"
