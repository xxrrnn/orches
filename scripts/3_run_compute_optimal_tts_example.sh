#!/usr/bin/env bash
# Run a reproducible Compute-Optimal-TTS smoke test from the uv environment.
#
# Common commands:
#   bash scripts/3_run_compute_optimal_tts_example.sh start-cot
#   TTS_RUN_ID=cot-seed0 TTS_SEED=0 bash scripts/3_run_compute_optimal_tts_example.sh run-cot
#   bash scripts/3_run_compute_optimal_tts_example.sh start-beam
#   TTS_RUN_ID=baseline-seed0 TTS_SEED=0 bash scripts/3_run_compute_optimal_tts_example.sh run-beam
#   bash scripts/3_run_compute_optimal_tts_example.sh status
#   bash scripts/3_run_compute_optimal_tts_example.sh stop
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT_DIR/environments/tts"
SOURCE_DIR="$ROOT_DIR/third_party/compute-optimal-tts/src"
PYTHON="$ENV_DIR/.venv/bin/python"

HOST_ADDR="${TTS_HOST_ADDR:-127.0.0.1}"
CONTROLLER_PORT="${TTS_CONTROLLER_PORT:-10014}"
POLICY_PORT="${TTS_POLICY_PORT:-10082}"
PRM_PORT="${TTS_PRM_PORT:-10081}"

POLICY_GPU="${TTS_POLICY_GPU:-0}"
PRM_GPU="${TTS_PRM_GPU:-0}"

POLICY_MODEL="${TTS_POLICY_MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
PRM_MODEL="${TTS_PRM_MODEL:-Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B}"
MAX_MODEL_LENGTH="${TTS_MAX_MODEL_LENGTH:-8192}"
MAX_NEW_TOKENS="${TTS_MAX_NEW_TOKENS:-4096}"
POLICY_GPU_MEMORY_UTILIZATION="${TTS_POLICY_GPU_MEMORY_UTILIZATION:-0.35}"
TTS_TEMPERATURE="${TTS_TEMPERATURE:-0.7}"
TTS_SEED="${TTS_SEED:-0}"
TTS_STRICT_DETERMINISM="${TTS_STRICT_DETERMINISM:-1}"
TTS_POLICY_MAX_CONCURRENCY="${TTS_POLICY_MAX_CONCURRENCY:-1}"
TTS_PRM_MAX_CONCURRENCY="${TTS_PRM_MAX_CONCURRENCY:-1}"
TTS_RUN_ID="${TTS_RUN_ID:-}"
TTS_LOCAL="${TTS_LOCAL:-1}"
TTS_TASK_NAME="${TTS_TASK_NAME:-AIME24}"
TTS_BEAM_SIZE="${TTS_BEAM_SIZE:-1}"
TTS_TREE_MAX_WIDTH="${TTS_TREE_MAX_WIDTH:-4}"
TTS_TREE_MAX_DEPTH="${TTS_TREE_MAX_DEPTH:-4}"
TTS_QUESTION_MAX_NUM="${TTS_QUESTION_MAX_NUM:-3}"
if [[ -n "${TTS_BATCH_SIZE:-}" ]]; then
    TTS_BATCH_SIZE="$TTS_BATCH_SIZE"
elif (( TTS_QUESTION_MAX_NUM > 0 )); then
    TTS_BATCH_SIZE="$TTS_QUESTION_MAX_NUM"
else
    TTS_BATCH_SIZE=30
fi
TTS_MAX_TIME="${TTS_MAX_TIME:-0}"
TTS_VERIFY_DETERMINISM="${TTS_VERIFY_DETERMINISM:-1}"
TTS_DETERMINISM_FIXTURE="${TTS_DETERMINISM_FIXTURE:-$ROOT_DIR/traces/Qwen1.5/Skywork-1.5B/AIME24/b1/w4/seed0/fixture-20260801-run4/sw/problem_0000.json}"

SAVE_BASE_DIR="${TTS_SAVE_BASE_DIR:-/tmp/orches-tts-runs}"
TRACE_BASE_DIR="${TTS_TRACE_BASE_DIR:-$ROOT_DIR/traces}"
SAVE_DIR="${TTS_SAVE_DIR:-}"
HF_HOME="${HF_HOME:-$ROOT_DIR/models}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

COMMAND="${1:-cot}"

if [[ ! -x "$PYTHON" ]]; then
    printf 'error: missing uv environment: %s\n' "$PYTHON" >&2
    printf 'run first: bash scripts/1_env_compute_optimal_tts.sh\n' >&2
    exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    printf 'error: missing Compute-Optimal-TTS source: %s\n' "$SOURCE_DIR" >&2
    printf 'run first: bash scripts/0_setup.sh\n' >&2
    exit 1
fi

export PYTHONPATH="$SOURCE_DIR"
export LOGDIR="$SOURCE_DIR/logs_fastchat"
export HF_HOME
export HF_HUB_CACHE
export HF_ENDPOINT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-$TTS_SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-0}"
export ORCHES_TTS_SEED="$TTS_SEED"
export ORCHES_TTS_STRICT_DETERMINISM="$TTS_STRICT_DETERMINISM"
export ORCHES_TTS_SOURCE_REVISION="${ORCHES_TTS_SOURCE_REVISION:-0ee2578af1f8d6cac445c9c4c72780528bb94556}"
export ORCHES_TTS_PATCH_SERIES="${ORCHES_TTS_PATCH_SERIES:-001-tts-rtx50-compatibility,002-tts-baseline-tracing,003-vllm-multisample-aggregation,004-compact-output-token-text,005-reward-score-alias,006-software-candidate-token-lengths,007-tts-strict-determinism}"
export ORCHES_RAY_LOCAL_IP="${ORCHES_RAY_LOCAL_IP:-127.0.0.1}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,0.0.0.0"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,0.0.0.0"

mkdir -p "$LOGDIR"

prepare_trace_run() {
    [[ -n "$TTS_RUN_ID" ]] || {
        printf 'error: set TTS_RUN_ID for a traceable, reproducible run (for example: baseline-aime24-seed0)\n' >&2
        exit 1
    }
    export ORCHES_TTS_SEED="$TTS_SEED"
    export ORCHES_TTS_RUN_ID="$TTS_RUN_ID"
    SAVE_DIR="${SAVE_DIR:-$SAVE_BASE_DIR/$TTS_RUN_ID/results}"
    [[ ! -e "$SAVE_DIR" ]] || {
        printf 'error: result directory already exists: %s\n' "$SAVE_DIR" >&2
        printf 'choose a new TTS_RUN_ID; existing results are never silently reused.\n' >&2
        exit 1
    }
    export ORCHES_TTS_TRACE_DIR="${ORCHES_TTS_TRACE_DIR:-$TRACE_BASE_DIR/$TTS_RUN_ID}"
    [[ ! -e "$ORCHES_TTS_TRACE_DIR" ]] || {
        printf 'error: trace directory already exists: %s\n' "$ORCHES_TTS_TRACE_DIR" >&2
        printf 'choose a new TTS_RUN_ID; existing traces are never silently reused.\n' >&2
        exit 1
    }
    mkdir -p "$SAVE_DIR"
}

validate_beam_configuration() {
    [[ "$TTS_BEAM_SIZE" =~ ^[1-9][0-9]*$ ]] || {
        printf 'error: TTS_BEAM_SIZE must be a positive integer: %s\n' "$TTS_BEAM_SIZE" >&2
        exit 1
    }
    [[ "$TTS_TREE_MAX_WIDTH" =~ ^[1-9][0-9]*$ ]] || {
        printf 'error: TTS_TREE_MAX_WIDTH must be a positive integer: %s\n' "$TTS_TREE_MAX_WIDTH" >&2
        exit 1
    }
    [[ "$TTS_TREE_MAX_DEPTH" =~ ^[1-9][0-9]*$ ]] || {
        printf 'error: TTS_TREE_MAX_DEPTH must be a positive integer: %s\n' "$TTS_TREE_MAX_DEPTH" >&2
        exit 1
    }
    (( TTS_TREE_MAX_WIDTH % TTS_BEAM_SIZE == 0 )) || {
        printf 'error: TTS_TREE_MAX_WIDTH (%s) must be divisible by TTS_BEAM_SIZE (%s)\n' \
            "$TTS_TREE_MAX_WIDTH" "$TTS_BEAM_SIZE" >&2
        exit 1
    }
}

verify_determinism() {
    [[ "$TTS_STRICT_DETERMINISM" == "1" ]] || return 0
    [[ -f "$TTS_DETERMINISM_FIXTURE" ]] || {
        printf 'error: determinism fixture is missing: %s\n' "$TTS_DETERMINISM_FIXTURE" >&2
        exit 1
    }
    "$PYTHON" "$ROOT_DIR/scripts/5_verify_tts_determinism.py" \
        --fixture "$TTS_DETERMINISM_FIXTURE" \
        --policy-address "http://$HOST_ADDR:$POLICY_PORT" \
        --prm-address "http://$HOST_ADDR:$PRM_PORT"
}

if [[ "$COMMAND" == "controller" ]]; then
    cd "$SOURCE_DIR"
    exec "$PYTHON" -m fastchat.serve.controller \
        --host "$HOST_ADDR" \
        --port "$CONTROLLER_PORT"
fi

if [[ "$COMMAND" == "policy-worker" ]]; then
    cd "$SOURCE_DIR"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
    export VLLM_HOST_IP="$HOST_ADDR"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export CUDA_VISIBLE_DEVICES="$POLICY_GPU"
    exec "$PYTHON" -m reason.llm_service.workers.vllm_worker \
        --max_model_length "$MAX_MODEL_LENGTH" \
        --gpu_memory_utilization "$POLICY_GPU_MEMORY_UTILIZATION" \
        --limit-worker-concurrency "$TTS_POLICY_MAX_CONCURRENCY" \
        --swap_space 16 \
        --model-path "$POLICY_MODEL" \
        --controller-address "http://$HOST_ADDR:$CONTROLLER_PORT" \
        --host "$HOST_ADDR" \
        --port "$POLICY_PORT" \
        --worker-address "http://$HOST_ADDR:$POLICY_PORT"
fi

if [[ "$COMMAND" == "prm-worker" ]]; then
    cd "$SOURCE_DIR"
    export CUDA_VISIBLE_DEVICES="$PRM_GPU"
    exec "$PYTHON" -m reason.llm_service.workers.reward_model_worker \
        --model-path "$PRM_MODEL" \
        --limit-worker-concurrency "$TTS_PRM_MAX_CONCURRENCY" \
        --seed "$TTS_SEED" \
        --controller-address "http://$HOST_ADDR:$CONTROLLER_PORT" \
        --host "$HOST_ADDR" \
        --port "$PRM_PORT" \
        --worker-address "http://$HOST_ADDR:$PRM_PORT"
fi

if [[ "$COMMAND" == "start-cot" || "$COMMAND" == "start-beam" ]]; then
    # tmux keeps a server-wide environment.  Set the run-specific values there
    # explicitly so detached workers receive the requested model pair and GPU
    # assignments rather than values from an earlier tmux server.
    tmux set-environment -g TTS_POLICY_MODEL "$POLICY_MODEL"
    tmux set-environment -g TTS_PRM_MODEL "$PRM_MODEL"
    tmux set-environment -g TTS_POLICY_GPU "$POLICY_GPU"
    tmux set-environment -g TTS_PRM_GPU "$PRM_GPU"
    tmux set-environment -g TTS_POLICY_GPU_MEMORY_UTILIZATION "$POLICY_GPU_MEMORY_UTILIZATION"
    tmux set-environment -g TTS_MAX_MODEL_LENGTH "$MAX_MODEL_LENGTH"
    for variable_name in TTS_SEED TTS_STRICT_DETERMINISM TTS_POLICY_MAX_CONCURRENCY TTS_PRM_MAX_CONCURRENCY HF_HOME HF_HUB_CACHE HF_ENDPOINT HF_HUB_OFFLINE PYTHONHASHSEED CUBLAS_WORKSPACE_CONFIG CUDA_DEVICE_MAX_CONNECTIONS NVIDIA_TF32_OVERRIDE ORCHES_TTS_SEED ORCHES_TTS_STRICT_DETERMINISM ORCHES_RAY_LOCAL_IP ORCHES_TTS_SOURCE_REVISION ORCHES_TTS_PATCH_SERIES; do
        tmux set-environment -g "$variable_name" "${!variable_name}"
    done
    tmux has-session -t tts-controller 2>/dev/null ||
        tmux new-session -d -s tts-controller "bash '$ROOT_DIR/scripts/3_run_compute_optimal_tts_example.sh' controller"

    tmux has-session -t tts-policy 2>/dev/null ||
        tmux new-session -d -s tts-policy "bash '$ROOT_DIR/scripts/3_run_compute_optimal_tts_example.sh' policy-worker"

    if [[ "$COMMAND" == "start-beam" ]]; then
        tmux has-session -t tts-prm 2>/dev/null ||
            tmux new-session -d -s tts-prm "bash '$ROOT_DIR/scripts/3_run_compute_optimal_tts_example.sh' prm-worker"
    fi

    printf 'Started tmux services. Wait until the workers finish loading, then run:\n'
    if [[ "$COMMAND" == "start-beam" ]]; then
        printf '  TTS_RUN_ID=baseline-seed0 TTS_SEED=0 bash scripts/3_run_compute_optimal_tts_example.sh run-beam\n'
    else
        printf '  TTS_RUN_ID=cot-seed0 TTS_SEED=0 bash scripts/3_run_compute_optimal_tts_example.sh run-cot\n'
    fi
    printf '\nUseful checks:\n'
    printf '  bash scripts/3_run_compute_optimal_tts_example.sh status\n'
    printf '  bash scripts/3_run_compute_optimal_tts_example.sh logs\n'
    exit 0
fi

if [[ "$COMMAND" == "run-cot" || "$COMMAND" == "cot" ]]; then
    [[ "$TTS_VERIFY_DETERMINISM" == "1" ]] && verify_determinism
    prepare_trace_run
    cd "$SOURCE_DIR"
    exec "$PYTHON" reason/evaluation/evaluate.py \
        --LM "$POLICY_MODEL" \
        --RM dummy \
        --task_name "$TTS_TASK_NAME" \
        --temperature "$TTS_TEMPERATURE" \
        --seed "$TTS_SEED" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --num_sequence 1 \
        --tree_max_width 1 \
        --tree_max_depth 1 \
        --save_dir "$SAVE_DIR" \
        --method cot \
        --num_worker 1 \
        --controller_addr "http://$HOST_ADDR:$CONTROLLER_PORT" \
        --add_step_prompt \
        --question_parallel_num 1 \
        --question_max_num "$TTS_QUESTION_MAX_NUM" \
        --double_line_break 1 \
        --batch_size "$TTS_BATCH_SIZE" \
        --max_time "$TTS_MAX_TIME" \
        --local "$TTS_LOCAL"
fi

if [[ "$COMMAND" == "run-beam" || "$COMMAND" == "beam" ]]; then
    [[ "$TTS_VERIFY_DETERMINISM" == "1" ]] && verify_determinism
    prepare_trace_run
    validate_beam_configuration
    cd "$SOURCE_DIR"
    exec "$PYTHON" reason/evaluation/evaluate.py \
        --LM "$POLICY_MODEL" \
        --RM "$PRM_MODEL" \
        --task_name "$TTS_TASK_NAME" \
        --temperature "$TTS_TEMPERATURE" \
        --seed "$TTS_SEED" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --num_sequence "$TTS_BEAM_SIZE" \
        --tree_max_width "$TTS_TREE_MAX_WIDTH" \
        --tree_max_depth "$TTS_TREE_MAX_DEPTH" \
        --save_dir "$SAVE_DIR" \
        --method beam_search \
        --num_worker 1 \
        --controller_addr "http://$HOST_ADDR:$CONTROLLER_PORT" \
        --add_step_prompt \
        --question_parallel_num 1 \
        --question_max_num "$TTS_QUESTION_MAX_NUM" \
        --double_line_break 1 \
        --batch_size "$TTS_BATCH_SIZE" \
        --max_time "$TTS_MAX_TIME" \
        --local "$TTS_LOCAL"
fi

if [[ "$COMMAND" == "verify-determinism" ]]; then
    verify_determinism
    exit 0
fi

if [[ "$COMMAND" == "status" ]]; then
    tmux ls 2>/dev/null | grep -E '^tts-' || true
    printf '\nController models:\n'
    curl --silent --show-error \
        --request POST \
        --header 'Content-Type: application/json' \
        --data '{}' \
        "http://$HOST_ADDR:$CONTROLLER_PORT/list_models" || true
    printf '\n'
    exit 0
fi

if [[ "$COMMAND" == "logs" ]]; then
    for session in tts-controller tts-policy tts-prm; do
        if tmux has-session -t "$session" 2>/dev/null; then
            printf '\n===== %s =====\n' "$session"
            tmux capture-pane -pt "$session:0" -S -80
        fi
    done
    exit 0
fi

if [[ "$COMMAND" == "stop" ]]; then
    tmux kill-session -t tts-controller 2>/dev/null || true
    tmux kill-session -t tts-policy 2>/dev/null || true
    tmux kill-session -t tts-prm 2>/dev/null || true
    printf 'Stopped TTS demo tmux sessions.\n'
    exit 0
fi

printf 'error: unknown command: %s\n' "$COMMAND" >&2
printf 'valid commands: start-cot, run-cot, start-beam, run-beam, status, logs, stop\n' >&2
exit 1
