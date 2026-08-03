#!/usr/bin/env bash
# Run the reproducible Llama-3.2-1B + Skywork-1.5B AIME24 beam/width grid.
#
# Default grid:
#   b1/w4, b1/w8, b2/w4, b2/w8
# Override with, for example:
#   TTS_GRID_CONFIGS="1/4 2/8" bash scripts/4_run_llama_skywork_aime_grid.sh run
#
# The run command checks the local Llama cache, restarts TTS services when the
# registered models do not match, then runs configs sequentially.  Do not run
# these configs in parallel when comparing byte-for-byte traces.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT_DIR/scripts/3_run_compute_optimal_tts_example.sh"
PYTHON="$ROOT_DIR/environments/tts/.venv/bin/python"

POLICY_MODEL="${TTS_POLICY_MODEL:-$ROOT_DIR/models/Llama-3.2-1B-Instruct}"
PRM_MODEL="${TTS_PRM_MODEL:-Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B}"
POLICY_LABEL="${TTS_POLICY_LABEL:-Llama3.2-1B}"
PRM_LABEL="${TTS_PRM_LABEL:-Skywork-1.5B}"
TASK_NAME="${TTS_TASK_NAME:-AIME24}"
RUN_STAMP="${TTS_RUN_STAMP:-$(date -u +%Y%m%d-aime24-seed${TTS_SEED:-0}-max${TTS_MAX_NEW_TOKENS:-8192})}"
COMMAND="${1:-run}"
POLICY_GPU="${TTS_POLICY_GPU:-0}"
PRM_GPU="${TTS_PRM_GPU:-0}"
MAX_MODEL_LENGTH="${TTS_MAX_MODEL_LENGTH:-16384}"
MAX_NEW_TOKENS="${TTS_MAX_NEW_TOKENS:-8192}"
TREE_MAX_DEPTH="${TTS_TREE_MAX_DEPTH:-4}"
QUESTION_MAX_NUM="${TTS_QUESTION_MAX_NUM:-0}"
BATCH_SIZE="${TTS_BATCH_SIZE:-30}"
MAX_TIME="${TTS_MAX_TIME:-0}"
SEED="${TTS_SEED:-0}"
STRICT_DETERMINISM="${TTS_STRICT_DETERMINISM:-1}"
VERIFY_DETERMINISM="${TTS_VERIFY_DETERMINISM:-0}"
POLICY_MAX_CONCURRENCY="${TTS_POLICY_MAX_CONCURRENCY:-1}"
PRM_MAX_CONCURRENCY="${TTS_PRM_MAX_CONCURRENCY:-1}"
GRID_CONFIGS="${TTS_GRID_CONFIGS:-1/4 1/8 2/4 2/8}"

if [[ -n "${TTS_POLICY_GPU_MEMORY_UTILIZATION:-}" ]]; then
    POLICY_GPU_MEMORY_UTILIZATION="$TTS_POLICY_GPU_MEMORY_UTILIZATION"
elif [[ "$POLICY_GPU" == "$PRM_GPU" ]]; then
    # Llama-1B and Skywork-1.5B fit together on a 32GB RTX 5090.  0.76 gives
    # vLLM an aggressive KV cache while leaving room for the PRM and CUDA
    # overhead.  If the policy worker OOMs, rerun with:
    #   TTS_POLICY_GPU_MEMORY_UTILIZATION=0.72 bash scripts/4_run_llama_skywork_aime_grid.sh start
    POLICY_GPU_MEMORY_UTILIZATION="0.76"
else
    POLICY_GPU_MEMORY_UTILIZATION="0.92"
fi

export HF_HOME="${HF_HOME:-$ROOT_DIR/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,0.0.0.0"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,0.0.0.0"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

validate_common_config() {
    [[ -x "$RUNNER" ]] || die "missing runner: $RUNNER"
    is_positive_integer "$MAX_MODEL_LENGTH" || die "TTS_MAX_MODEL_LENGTH must be a positive integer: $MAX_MODEL_LENGTH"
    is_positive_integer "$MAX_NEW_TOKENS" || die "TTS_MAX_NEW_TOKENS must be a positive integer: $MAX_NEW_TOKENS"
    is_positive_integer "$TREE_MAX_DEPTH" || die "TTS_TREE_MAX_DEPTH must be a positive integer: $TREE_MAX_DEPTH"
    [[ "$QUESTION_MAX_NUM" =~ ^[0-9]+$ ]] || die "TTS_QUESTION_MAX_NUM must be a non-negative integer: $QUESTION_MAX_NUM"
    is_positive_integer "$BATCH_SIZE" || die "TTS_BATCH_SIZE must be a positive integer: $BATCH_SIZE"
    [[ "$MAX_TIME" =~ ^[0-9]+$ ]] || die "TTS_MAX_TIME must be a non-negative integer: $MAX_TIME"
    [[ "$POLICY_GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$|^1(\.0+)?$ ]] ||
        die "TTS_POLICY_GPU_MEMORY_UTILIZATION must be between 0 and 1: $POLICY_GPU_MEMORY_UTILIZATION"
}

check_llama_cache() {
    [[ -x "$PYTHON" ]] || die "missing TTS uv environment: $PYTHON; run first: bash scripts/1_env_compute_optimal_tts.sh"
    [[ -e "$POLICY_MODEL" || "$POLICY_MODEL" != /* ]] || die "missing local Llama policy path: $POLICY_MODEL"

    "$PYTHON" - "$POLICY_MODEL" <<'PY'
import sys
from transformers import AutoConfig, AutoTokenizer

model = sys.argv[1]
config = AutoConfig.from_pretrained(model, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
if config.model_type != "llama":
    raise SystemExit(f"expected a Llama model, got model_type={config.model_type}")
print(f"Llama cache OK: {model}")
print(f"  model_type={config.model_type}")
print(f"  max_position_embeddings={getattr(config, 'max_position_embeddings', None)}")
print(f"  tokenizer={tokenizer.__class__.__name__}")
PY
}

service_status() {
    TTS_POLICY_MODEL="$POLICY_MODEL" TTS_PRM_MODEL="$PRM_MODEL" bash "$RUNNER" status || true
}

services_match() {
    status="$1"
    grep -F "$PRM_MODEL" <<<"$status" >/dev/null &&
        grep -F "$POLICY_MODEL" <<<"$status" >/dev/null
}

tmux_env_value() {
    tmux show-environment -g "$1" 2>/dev/null | sed "s/^$1=//"
}

runtime_config_matches() {
    [[ "$(tmux_env_value TTS_POLICY_GPU)" == "$POLICY_GPU" ]] &&
        [[ "$(tmux_env_value TTS_PRM_GPU)" == "$PRM_GPU" ]] &&
        [[ "$(tmux_env_value TTS_MAX_MODEL_LENGTH)" == "$MAX_MODEL_LENGTH" ]] &&
        [[ "$(tmux_env_value TTS_POLICY_GPU_MEMORY_UTILIZATION)" == "$POLICY_GPU_MEMORY_UTILIZATION" ]]
}

check_services() {
    status="$(TTS_POLICY_MODEL="$POLICY_MODEL" TTS_PRM_MODEL="$PRM_MODEL" bash "$RUNNER" status)"
    printf '%s\n' "$status"
    services_match "$status" || die "TTS services are not registered for the requested Llama/Skywork model pair"
    runtime_config_matches || die "TTS services are registered, but tmux runtime config does not match this script"
}

start_services() {
    check_llama_cache

    status="$(service_status)"
    if services_match "$status" && runtime_config_matches; then
        printf '%s\n' "$status"
        printf 'Services already match requested model pair.\n'
        return 0
    fi

    printf 'Restarting TTS services for policy=%s prm=%s\n' "$POLICY_MODEL" "$PRM_MODEL"
    bash "$RUNNER" stop
    TTS_POLICY_MODEL="$POLICY_MODEL" \
    TTS_PRM_MODEL="$PRM_MODEL" \
    TTS_POLICY_GPU="$POLICY_GPU" \
    TTS_PRM_GPU="$PRM_GPU" \
    TTS_POLICY_GPU_MEMORY_UTILIZATION="$POLICY_GPU_MEMORY_UTILIZATION" \
    TTS_MAX_MODEL_LENGTH="$MAX_MODEL_LENGTH" \
    TTS_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    TTS_STRICT_DETERMINISM="$STRICT_DETERMINISM" \
    TTS_POLICY_MAX_CONCURRENCY="$POLICY_MAX_CONCURRENCY" \
    TTS_PRM_MAX_CONCURRENCY="$PRM_MAX_CONCURRENCY" \
        bash "$RUNNER" start-beam

    for attempt in $(seq 1 60); do
        status="$(service_status)"
        if services_match "$status"; then
            printf '%s\n' "$status"
            printf 'Services ready after %s checks.\n' "$attempt"
            return 0
        fi
        sleep 5
    done

    bash "$RUNNER" logs || true
    die "timed out waiting for Llama/Skywork services to register"
}

run_config() {
    beam="$1"
    width="$2"
    is_positive_integer "$beam" || die "beam must be a positive integer: $beam"
    is_positive_integer "$width" || die "width must be a positive integer: $width"
    (( width % beam == 0 )) || die "width ($width) must be divisible by beam ($beam)"

    run_id="$POLICY_LABEL/$PRM_LABEL/$TASK_NAME/b$beam/w$width/seed${SEED}/$RUN_STAMP"
    save_dir="${TTS_SAVE_BASE_DIR:-/tmp/orches-tts-runs}/$run_id/results"
    trace_dir="${TTS_TRACE_BASE_DIR:-$ROOT_DIR/traces}/$run_id"
    [[ ! -e "$save_dir" ]] || die "result directory already exists: $save_dir"
    [[ ! -e "$trace_dir" ]] || die "trace directory already exists: $trace_dir"

    printf '\n===== RUN %s =====\n' "$run_id"
    TTS_POLICY_MODEL="$POLICY_MODEL" \
    TTS_PRM_MODEL="$PRM_MODEL" \
    TTS_RUN_ID="$run_id" \
    TTS_TASK_NAME="$TASK_NAME" \
    TTS_BEAM_SIZE="$beam" \
    TTS_TREE_MAX_WIDTH="$width" \
    TTS_TREE_MAX_DEPTH="$TREE_MAX_DEPTH" \
    TTS_QUESTION_MAX_NUM="$QUESTION_MAX_NUM" \
    TTS_BATCH_SIZE="$BATCH_SIZE" \
    TTS_MAX_TIME="$MAX_TIME" \
    TTS_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    TTS_MAX_MODEL_LENGTH="$MAX_MODEL_LENGTH" \
    TTS_SEED="$SEED" \
    TTS_STRICT_DETERMINISM="$STRICT_DETERMINISM" \
    TTS_VERIFY_DETERMINISM="$VERIFY_DETERMINISM" \
    TTS_POLICY_GPU="$POLICY_GPU" \
    TTS_PRM_GPU="$PRM_GPU" \
    TTS_POLICY_GPU_MEMORY_UTILIZATION="$POLICY_GPU_MEMORY_UTILIZATION" \
    TTS_POLICY_MAX_CONCURRENCY="$POLICY_MAX_CONCURRENCY" \
    TTS_PRM_MAX_CONCURRENCY="$PRM_MAX_CONCURRENCY" \
    HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
        bash "$RUNNER" run-beam
    printf '===== DONE %s =====\n' "$run_id"
}

run_grid() {
    for config in $GRID_CONFIGS; do
        [[ "$config" =~ ^[1-9][0-9]*/[1-9][0-9]*$ ]] ||
            die "invalid TTS_GRID_CONFIGS entry: $config; expected beam/width"
        run_config "${config%%/*}" "${config##*/}"
    done
}

case "$COMMAND" in
    check)
        validate_common_config
        check_llama_cache
        check_services
        ;;
    start)
        validate_common_config
        start_services
        ;;
    run)
        validate_common_config
        check_llama_cache
        start_services
        run_grid
        ;;
    status)
        TTS_POLICY_MODEL="$POLICY_MODEL" TTS_PRM_MODEL="$PRM_MODEL" bash "$RUNNER" status
        ;;
    logs)
        bash "$RUNNER" logs
        ;;
    stop)
        bash "$RUNNER" stop
        ;;
    *)
        printf 'usage: %s {check|start|run|status|logs|stop}\n' "${BASH_SOURCE[0]}" >&2
        exit 1
        ;;
esac
