#!/usr/bin/env bash
# Run a small Compute-Optimal-TTS smoke test from the uv environment.
#
# Common commands:
#   bash scripts/3_run_compute_optimal_tts_example.sh start-cot
#   bash scripts/3_run_compute_optimal_tts_example.sh run-cot
#   bash scripts/3_run_compute_optimal_tts_example.sh start-beam
#   bash scripts/3_run_compute_optimal_tts_example.sh run-beam
#   bash scripts/3_run_compute_optimal_tts_example.sh status
#   bash scripts/3_run_compute_optimal_tts_example.sh stop
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TTS_TARGET:-rtx5090}"
ENV_DIR="$ROOT_DIR/environments/compute-optimal-tts/$TARGET"
SOURCE_DIR="$ROOT_DIR/third_party/compute-optimal-tts/src"
PYTHON="$ENV_DIR/.venv/bin/python"

HOST_ADDR="${TTS_HOST_ADDR:-127.0.0.1}"
CONTROLLER_PORT="${TTS_CONTROLLER_PORT:-10014}"
POLICY_PORT="${TTS_POLICY_PORT:-10082}"
PRM_PORT="${TTS_PRM_PORT:-10081}"

POLICY_GPU="${TTS_POLICY_GPU:-0}"
PRM_GPU="${TTS_PRM_GPU:-1}"

POLICY_MODEL="${TTS_POLICY_MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
PRM_MODEL="${TTS_PRM_MODEL:-Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B}"
MAX_MODEL_LENGTH="${TTS_MAX_MODEL_LENGTH:-4096}"
MAX_NEW_TOKENS="${TTS_MAX_NEW_TOKENS:-1024}"

SAVE_DIR="${TTS_SAVE_DIR:-/tmp/orches-tts-demo}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface/orches-tts}"
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
export HF_HUB_CACHE="$HF_HOME"
export HF_ENDPOINT
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,0.0.0.0"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,0.0.0.0"

mkdir -p "$LOGDIR" "$SAVE_DIR"

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
        --gpu_memory_utilization 0.88 \
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
        --controller-address "http://$HOST_ADDR:$CONTROLLER_PORT" \
        --host "$HOST_ADDR" \
        --port "$PRM_PORT" \
        --worker-address "http://$HOST_ADDR:$PRM_PORT"
fi

if [[ "$COMMAND" == "start-cot" || "$COMMAND" == "start-beam" ]]; then
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
        printf '  bash scripts/3_run_compute_optimal_tts_example.sh run-beam\n'
    else
        printf '  bash scripts/3_run_compute_optimal_tts_example.sh run-cot\n'
    fi
    printf '\nUseful checks:\n'
    printf '  bash scripts/3_run_compute_optimal_tts_example.sh status\n'
    printf '  bash scripts/3_run_compute_optimal_tts_example.sh logs\n'
    exit 0
fi

if [[ "$COMMAND" == "run-cot" || "$COMMAND" == "cot" ]]; then
    cd "$SOURCE_DIR"
    exec "$PYTHON" reason/evaluation/evaluate.py \
        --LM "$POLICY_MODEL" \
        --RM dummy \
        --task_name AIME24 \
        --temperature 0.7 \
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
        --double_line_break 1 \
        --batch_size 1 \
        --max_time 1 \
        --local 0
fi

if [[ "$COMMAND" == "run-beam" || "$COMMAND" == "beam" ]]; then
    cd "$SOURCE_DIR"
    exec "$PYTHON" reason/evaluation/evaluate.py \
        --LM "$POLICY_MODEL" \
        --RM "$PRM_MODEL" \
        --task_name AIME24 \
        --temperature 0.7 \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --num_sequence 1 \
        --tree_max_width 2 \
        --tree_max_depth 4 \
        --save_dir "$SAVE_DIR" \
        --method beam_search \
        --num_worker 1 \
        --controller_addr "http://$HOST_ADDR:$CONTROLLER_PORT" \
        --add_step_prompt \
        --question_parallel_num 1 \
        --double_line_break 1 \
        --batch_size 1 \
        --max_time 1 \
        --local 0
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
