#!/usr/bin/env bash
# Generate a small, reproducible Qwen-1.5B + Skywork-1.5B beam-search trace.
#
# Usage:
#   bash scripts/4_generate_qwen05_skywork_trace.sh start
#   bash scripts/4_generate_qwen05_skywork_trace.sh verify
#   bash scripts/4_generate_qwen05_skywork_trace.sh status
#   bash scripts/4_generate_qwen05_skywork_trace.sh run
#   bash scripts/4_generate_qwen05_skywork_trace.sh stop
#
# Override model IDs, task, beam, width, depth, or the short directory labels
# with the corresponding TTS_* environment variables before invoking this file.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT_DIR/scripts/3_run_compute_optimal_tts_example.sh"

POLICY_MODEL="${TTS_POLICY_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PRM_MODEL="${TTS_PRM_MODEL:-Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B}"
TASK_NAME="${TTS_TASK_NAME:-AIME24}"
BEAM_SIZE="${TTS_BEAM_SIZE:-2}"
TREE_MAX_WIDTH="${TTS_TREE_MAX_WIDTH:-4}"
TREE_MAX_DEPTH="${TTS_TREE_MAX_DEPTH:-4}"
QUESTION_MAX_NUM="${TTS_QUESTION_MAX_NUM:-1}"
MAX_NEW_TOKENS="${TTS_MAX_NEW_TOKENS:-64}"
POLICY_LABEL="${TTS_POLICY_LABEL:-Qwen1.5}"
PRM_LABEL="${TTS_PRM_LABEL:-Skywork-1.5B}"
COMMAND="${1:-run}"

[[ -x "$RUNNER" ]] || {
    printf 'error: missing runner: %s\n' "$RUNNER" >&2
    exit 1
}

[[ "$BEAM_SIZE" =~ ^[1-9][0-9]*$ && "$TREE_MAX_WIDTH" =~ ^[1-9][0-9]*$ && "$TREE_MAX_DEPTH" =~ ^[1-9][0-9]*$ ]] || {
    printf 'error: beam, width, and depth must be positive integers\n' >&2
    exit 1
}
(( TREE_MAX_WIDTH % BEAM_SIZE == 0 )) || {
    printf 'error: width (%s) must be divisible by beam (%s)\n' "$TREE_MAX_WIDTH" "$BEAM_SIZE" >&2
    exit 1
}

export TTS_POLICY_MODEL="$POLICY_MODEL"
export TTS_PRM_MODEL="$PRM_MODEL"
export TTS_TASK_NAME="$TASK_NAME"
export TTS_BEAM_SIZE="$BEAM_SIZE"
export TTS_TREE_MAX_WIDTH="$TREE_MAX_WIDTH"
export TTS_TREE_MAX_DEPTH="$TREE_MAX_DEPTH"
export TTS_QUESTION_MAX_NUM="$QUESTION_MAX_NUM"
export TTS_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
export TTS_TRACE_BASE_DIR="${TTS_TRACE_BASE_DIR:-$ROOT_DIR/traces}"
export TTS_RUN_ID="${TTS_RUN_ID:-$POLICY_LABEL/$PRM_LABEL/$TASK_NAME/b$BEAM_SIZE/w$TREE_MAX_WIDTH}"

case "$COMMAND" in
    start) exec bash "$RUNNER" start-beam ;;
    verify) exec bash "$RUNNER" verify-determinism ;;
    run) exec bash "$RUNNER" run-beam ;;
    status) exec bash "$RUNNER" status ;;
    logs) exec bash "$RUNNER" logs ;;
    stop) exec bash "$RUNNER" stop ;;
    *)
        printf 'usage: %s {start|verify|run|status|logs|stop}\n' "${BASH_SOURCE[0]}" >&2
        exit 1
        ;;
esac
