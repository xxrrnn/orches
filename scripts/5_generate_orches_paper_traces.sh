#!/usr/bin/env bash
# Generate ORCHES paper text-pipeline traces (see Algorithm.md).
#
# Text trace matrix used for this reproduction run:
#   3 policies × 3 PRMs × widths {2,4,8} × beam {1}
#   default jobs: 9 × 3 = 27 full-MATH500 traces
#
# Bandwidth 100/75/50% is a simulator replay knob, not generated here.
#
# Search defaults follow Compute-Optimal-TTS [18] beam_search:
#   depth=40, max_new_tokens=8192, temperature=0.7, task=MATH
# This run uses beam=1 so paper "branch count" maps directly to tree_max_width.
# To run a conservative beam sweep later, pass TTS_BEAMS="1 2".
#
# GPU scheduling (defaults to physical GPU 2 and 3):
#   - every model pair is pinned to one physical GPU for reproducibility
#   - pairs are run from small to large PRM, with two independent GPU lanes
#
# Commands:
#   bash scripts/5_generate_orches_paper_traces.sh plan
#   bash scripts/5_generate_orches_paper_traces.sh check
#   bash scripts/5_generate_orches_paper_traces.sh run
#   bash scripts/5_generate_orches_paper_traces.sh status|logs|stop
#
# Useful overrides:
#   TTS_BEAMS="1 2" TTS_WIDTHS="2 4 8" bash scripts/5_generate_orches_paper_traces.sh run
#   TTS_BEAMS=1 TTS_QUESTION_MAX_NUM=3 bash scripts/5_generate_orches_paper_traces.sh run
#   TTS_PAIRS="Llama3.2-1B|Qwen2.5-1.5B-PRM" bash scripts/5_generate_orches_paper_traces.sh run
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT_DIR/scripts/3_run_compute_optimal_tts_example.sh"
PYTHON="$ROOT_DIR/environments/tts/.venv/bin/python"
COMMAND="${1:-plan}"

# --- paper matrix ------------------------------------------------------------
SEED="${TTS_SEED:-0}"
# Prefer TTS_BEAMS="1 2". TTS_BEAM_SIZE still works as a single-beam override.
if [[ -n "${TTS_BEAMS:-}" ]]; then
    BEAMS="$TTS_BEAMS"
elif [[ -n "${TTS_BEAM_SIZE:-}" ]]; then
    BEAMS="$TTS_BEAM_SIZE"
else
    BEAMS="1"
fi
TREE_MAX_DEPTH="${TTS_TREE_MAX_DEPTH:-40}"
MAX_NEW_TOKENS="${TTS_MAX_NEW_TOKENS:-8192}"
# Paper traces intentionally pin the model context budget.  Use the paper
# specific override below; do not inherit a generic shell TTS_MAX_MODEL_LENGTH
# from earlier smoke tests or interactive runs.
MAX_MODEL_LENGTH="${TTS_PAPER_MAX_MODEL_LENGTH:-8192}"
TEMPERATURE="${TTS_TEMPERATURE:-0.7}"
TASK_NAME="${TTS_TASK_NAME:-MATH}"
TASK_LABEL="${TTS_TASK_LABEL:-MATH500}"
QUESTION_MAX_NUM="${TTS_QUESTION_MAX_NUM:-0}"
BATCH_SIZE="${TTS_BATCH_SIZE:-500}"
MAX_TIME="${TTS_MAX_TIME:-0}"
STRICT_DETERMINISM="${TTS_STRICT_DETERMINISM:-1}"
# Paper pairs do not share the Qwen1.5 fixture; keep strict mode, skip fixture ping.
VERIFY_DETERMINISM="${TTS_VERIFY_DETERMINISM:-0}"
POLICY_MAX_CONCURRENCY="${TTS_POLICY_MAX_CONCURRENCY:-1}"
PRM_MAX_CONCURRENCY="${TTS_PRM_MAX_CONCURRENCY:-1}"
PAPER_POLICY_GPU_MEMORY_UTILIZATION="${TTS_PAPER_POLICY_GPU_MEMORY_UTILIZATION:-}"
RESUME="${TTS_RESUME:-1}"
# Branch counts for this run.  Odd widths are skipped automatically if beam=2.
WIDTHS="${TTS_WIDTHS:-2 4 8}"
# Keep the paper trace stamp stable across midnight and shell restarts.  A
# date-based default makes resume switch to a new output tree on the next day.
RUN_STAMP="${TTS_RUN_STAMP:-20260802-math500-seed${SEED}-d${TREE_MAX_DEPTH}-max${MAX_NEW_TOKENS}}"

GPU_A="${TTS_GPU_A:-2}"
GPU_B="${TTS_GPU_B:-3}"
LANE0_CONTROLLER_PORT="${TTS_LANE0_CONTROLLER_PORT:-11014}"
LANE0_POLICY_PORT="${TTS_LANE0_POLICY_PORT:-11082}"
LANE0_PRM_PORT="${TTS_LANE0_PRM_PORT:-11081}"
LANE1_CONTROLLER_PORT="${TTS_LANE1_CONTROLLER_PORT:-12014}"
LANE1_POLICY_PORT="${TTS_LANE1_POLICY_PORT:-12082}"
LANE1_PRM_PORT="${TTS_LANE1_PRM_PORT:-12081}"

TRACE_BASE_DIR="${TTS_TRACE_BASE_DIR:-$ROOT_DIR/traces}"
SAVE_BASE_DIR="${TTS_SAVE_BASE_DIR:-/tmp/orches-tts-runs}"
LOG_DIR="${TTS_PAPER_LOG_DIR:-$ROOT_DIR/logs/orches-paper-traces}"
RUN_LOCK_DIR="${TTS_PAPER_RUN_LOCK_DIR:-$ROOT_DIR/.orches-paper-traces.lock}"

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

model_weights_ready() {
    local path="$1"
    [[ -f "$path/config.json" ]] || return 1
    [[ -f "$path/model.safetensors" || -f "$path/pytorch_model.bin" || -f "$path/model.safetensors.index.json" || -f "$path/pytorch_model.bin.index.json" ]]
}

model_path() {
    printf '%s/models/%s\n' "$ROOT_DIR" "$1"
}

# Paper PRM labels -> concrete weights available in this repo / download.sh.
# Paper names are *-PRM-Tuned; Algorithm.md forbids guessing undisclosed forks,
# so we bind to the Compute-Optimal-TTS / download.sh stand-ins and record them.
POLICY_SPECS=(
    "Llama3.2-1B|Llama-3.2-1B-Instruct"
    "Qwen2.5-1.5B|Qwen2.5-1.5B-Instruct"
    "Qwen2.5-3B|Qwen2.5-3B-Instruct"
)

PRM_SPECS=(
    # label|local_dir|size_class(small|large)
    "Qwen2.5-1.5B-PRM|Skywork-o1-Open-PRM-Qwen-2.5-1.5B|small"
    "Qwen2.5-7B-PRM|Skywork-o1-Open-PRM-Qwen-2.5-7B|large"
    "Llama3.1-8B-PRM|Llama3.1-8B-PRM-Mistral-Data|large"
)

lookup_policy() {
    local label="$1" spec
    for spec in "${POLICY_SPECS[@]}"; do
        if [[ "${spec%%|*}" == "$label" ]]; then
            printf '%s\n' "$spec"
            return 0
        fi
    done
    return 1
}

lookup_prm() {
    local label="$1" spec
    for spec in "${PRM_SPECS[@]}"; do
        if [[ "${spec%%|*}" == "$label" ]]; then
            printf '%s\n' "$spec"
            return 0
        fi
    done
    return 1
}

default_pairs() {
    local p_spec r_spec p_label r_label
    # Run from small to large by PRM size first, then policy size.
    for r_spec in "${PRM_SPECS[@]}"; do
        r_label="${r_spec%%|*}"
        for p_spec in "${POLICY_SPECS[@]}"; do
            p_label="${p_spec%%|*}"
            printf '%s|%s\n' "$p_label" "$r_label"
        done
    done
}

requested_pairs() {
    if [[ -n "${TTS_PAIRS:-}" ]]; then
        # shellcheck disable=SC2086
        printf '%s\n' $TTS_PAIRS
    else
        default_pairs
    fi
}

# Widths that satisfy: width % beam == 0
widths_for_beam() {
    local beam="$1" width
    for width in $WIDTHS; do
        if (( width % beam == 0 )); then
            printf '%s ' "$width"
        fi
    done
}

run_id_for() {
    local policy_label="$1" prm_label="$2" beam="$3" width="$4"
    printf '%s/%s/%s/b%s/w%s/seed%s/%s' \
        "$policy_label" "$prm_label" "$TASK_LABEL" "$beam" "$width" "$SEED" "$RUN_STAMP"
}

trace_dir_for() {
    printf '%s/%s' "$TRACE_BASE_DIR" "$(run_id_for "$1" "$2" "$3" "$4")"
}

expected_problem_count() {
    if (( QUESTION_MAX_NUM > 0 )); then
        printf '%s\n' "$QUESTION_MAX_NUM"
    elif [[ "$TASK_LABEL" == "MATH500" ]]; then
        printf '500\n'
    else
        printf '%s\n' "$BATCH_SIZE"
    fi
}

config_done() {
    local dir count last_id
    dir="$(trace_dir_for "$1" "$2" "$3" "$4")"
    [[ -f "$dir/manifest.json" ]] || return 1
    count="$(expected_problem_count)"
    (( count > 0 )) || return 0
    last_id="$(printf 'problem_%04d.json' "$((count - 1))")"
    [[ -f "$dir/sw/$last_id" && -f "$dir/hw/$last_id" ]]
}

# True if this model pair still has any unfinished beam/width job.
pair_needs_run() {
    local policy_label="$1" prm_label="$2"
    local beam width widths
    for beam in $BEAMS; do
        widths="$(widths_for_beam "$beam")"
        for width in $widths; do
            if ! config_done "$policy_label" "$prm_label" "$beam" "$width"; then
                return 0
            fi
        done
    done
    return 1
}

validate_common() {
    local beam width widths found_any=0

    [[ -x "$RUNNER" ]] || die "missing runner: $RUNNER"
    [[ -x "$PYTHON" ]] || die "missing TTS uv environment: $PYTHON; run: bash scripts/1_env_compute_optimal_tts.sh"
    is_positive_integer "$TREE_MAX_DEPTH" || die "TTS_TREE_MAX_DEPTH must be a positive integer"
    is_positive_integer "$MAX_NEW_TOKENS" || die "TTS_MAX_NEW_TOKENS must be a positive integer"
    is_positive_integer "$MAX_MODEL_LENGTH" || die "TTS_MAX_MODEL_LENGTH must be a positive integer"
    is_positive_integer "$BATCH_SIZE" || die "TTS_BATCH_SIZE must be a positive integer"
    [[ "$QUESTION_MAX_NUM" =~ ^[0-9]+$ ]] || die "TTS_QUESTION_MAX_NUM must be a non-negative integer"
    [[ "$SEED" =~ ^[0-9]+$ ]] || die "TTS_SEED must be a non-negative integer"
    if [[ -n "$PAPER_POLICY_GPU_MEMORY_UTILIZATION" ]]; then
        [[ "$PAPER_POLICY_GPU_MEMORY_UTILIZATION" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || \
            die "TTS_PAPER_POLICY_GPU_MEMORY_UTILIZATION must be between 0 and 1"
    fi
    if [[ -n "${TTS_POLICY_GPU_MEMORY_UTILIZATION:-}" ]]; then
        printf 'note: ignoring generic TTS_POLICY_GPU_MEMORY_UTILIZATION=%s; paper run uses TTS_PAPER_POLICY_GPU_MEMORY_UTILIZATION or built-in safe defaults\n' \
            "$TTS_POLICY_GPU_MEMORY_UTILIZATION" >&2
    fi
    if [[ -n "${TTS_MAX_MODEL_LENGTH:-}" ]]; then
        printf 'note: ignoring generic TTS_MAX_MODEL_LENGTH=%s; paper run uses TTS_PAPER_MAX_MODEL_LENGTH=%s\n' \
            "$TTS_MAX_MODEL_LENGTH" "$MAX_MODEL_LENGTH" >&2
    fi

    for width in $WIDTHS; do
        is_positive_integer "$width" || die "invalid width: $width"
    done
    for beam in $BEAMS; do
        is_positive_integer "$beam" || die "invalid beam: $beam"
        widths="$(widths_for_beam "$beam")"
        [[ -n "${widths// /}" ]] || die "beam=$beam has no compatible width in: $WIDTHS"
        found_any=1
    done
    (( found_any )) || die "TTS_BEAMS is empty"
}

print_plan() {
    local policy_label prm_label beam width widths count=0 pending=0 done_n=0
    local p_spec r_spec policy_model prm_model prm_class

    printf 'ORCHES paper text-trace plan\n'
    printf '  task=%s (evaluator=%s)  beams=%s  depth=%s  max_new_tokens=%s  seed=%s\n' \
        "$TASK_LABEL" "$TASK_NAME" "$BEAMS" "$TREE_MAX_DEPTH" "$MAX_NEW_TOKENS" "$SEED"
    printf '  candidate widths=%s  (kept only when width %% beam == 0)\n' "$WIDTHS"
    printf '  GPUs: lane0=%s lane1=%s | pair-sticky same-GPU scheduling\n' \
        "$GPU_A" "$GPU_B"
    if (( QUESTION_MAX_NUM == 0 )); then
        printf '  problems=full %s  batch_size=%s\n' "$TASK_LABEL" "$BATCH_SIZE"
    else
        printf '  problems=first %s  batch_size=%s\n' "$QUESTION_MAX_NUM" "$BATCH_SIZE"
    fi
    printf '  run_stamp=%s\n' "$RUN_STAMP"
    printf '  resume=%s  strict_determinism=%s  verify_fixture=%s\n' \
        "$RESUME" "$STRICT_DETERMINISM" "$VERIFY_DETERMINISM"
    printf '\nBeam/width grid:\n'
    for beam in $BEAMS; do
        printf '  beam=%s -> %s\n' "$beam" "$(widths_for_beam "$beam")"
    done
    printf '\nJobs:\n'

    while IFS='|' read -r policy_label prm_label; do
        p_spec="$(lookup_policy "$policy_label")" || die "unknown policy label: $policy_label"
        r_spec="$(lookup_prm "$prm_label")" || die "unknown PRM label: $prm_label"
        IFS='|' read -r _ p_local <<<"$p_spec"
        IFS='|' read -r _ r_local prm_class <<<"$r_spec"
        policy_model="$(model_path "$p_local")"
        prm_model="$(model_path "$r_local")"
        for beam in $BEAMS; do
            widths="$(widths_for_beam "$beam")"
            for width in $widths; do
                count=$((count + 1))
                if config_done "$policy_label" "$prm_label" "$beam" "$width"; then
                    done_n=$((done_n + 1))
                    printf '  [done] %s  (%s)\n' \
                        "$(run_id_for "$policy_label" "$prm_label" "$beam" "$width")" "$prm_class"
                else
                    pending=$((pending + 1))
                    printf '  [todo] %s\n' \
                        "$(run_id_for "$policy_label" "$prm_label" "$beam" "$width")"
                    printf '         policy=%s\n' "$policy_model"
                    printf '         prm=%s  class=%s\n' "$prm_model" "$prm_class"
                fi
            done
        done
    done < <(requested_pairs)

    printf '\nTotal=%s  pending=%s  done=%s\n' "$count" "$pending" "$done_n"
    printf 'Note: bandwidth sweep (100/75/50%%) happens at simulator replay, not here.\n'
}

check_models() {
    local policy_label prm_label p_spec r_spec policy_model prm_model missing=0
    local -A seen=()

    printf 'Checking model weights (HF_HUB_OFFLINE=%s)\n' "$HF_HUB_OFFLINE"
    while IFS='|' read -r policy_label prm_label; do
        p_spec="$(lookup_policy "$policy_label")" || die "unknown policy label: $policy_label"
        r_spec="$(lookup_prm "$prm_label")" || die "unknown PRM label: $prm_label"
        IFS='|' read -r _ p_local <<<"$p_spec"
        IFS='|' read -r _ r_local _ <<<"$r_spec"
        policy_model="$(model_path "$p_local")"
        prm_model="$(model_path "$r_local")"

        if [[ -z "${seen[policy:$policy_label]:-}" ]]; then
            seen["policy:$policy_label"]=1
            if model_weights_ready "$policy_model"; then
                printf '  OK      policy %s -> %s\n' "$policy_label" "$policy_model"
            else
                printf '  MISSING policy %s -> %s\n' "$policy_label" "$policy_model"
                missing=1
            fi
        fi
        if [[ -z "${seen[prm:$prm_label]:-}" ]]; then
            seen["prm:$prm_label"]=1
            if model_weights_ready "$prm_model"; then
                printf '  OK      prm    %s -> %s\n' "$prm_label" "$prm_model"
            else
                printf '  MISSING prm    %s -> %s\n' "$prm_label" "$prm_model"
                missing=1
            fi
        fi
    done < <(requested_pairs)

    if (( missing )); then
        printf '\nDownload missing paper models, then re-run check:\n' >&2
        printf '  bash scripts/download.sh\n' >&2
        printf '  # or: bash models/download_paper_models.sh\n' >&2
        printf 'Subset example with currently cached weights:\n' >&2
        printf '  TTS_PAIRS="Llama3.2-1B|Qwen2.5-1.5B-PRM" TTS_BEAMS="1 2" TTS_WIDTHS="2 4" \\\n' >&2
        printf '    TTS_QUESTION_MAX_NUM=3 bash scripts/5_generate_orches_paper_traces.sh run\n' >&2
        exit 1
    fi
    printf 'All requested model weights look complete.\n'
}

stop_instance() {
    local instance="$1"
    local controller_port="$2"
    local policy_port="$3"
    local prm_port="$4"
    TTS_INSTANCE="$instance" \
    TTS_CONTROLLER_PORT="$controller_port" \
    TTS_POLICY_PORT="$policy_port" \
    TTS_PRM_PORT="$prm_port" \
        bash "$RUNNER" stop >/dev/null || true
}

stop_all_stacks() {
    stop_instance lane0 "$LANE0_CONTROLLER_PORT" "$LANE0_POLICY_PORT" "$LANE0_PRM_PORT"
    stop_instance lane1 "$LANE1_CONTROLLER_PORT" "$LANE1_POLICY_PORT" "$LANE1_PRM_PORT"
    # Also clear the legacy single-stack sessions if present.
    bash "$RUNNER" stop >/dev/null || true
    printf 'Stopped paper-trace TTS stacks (lane0/lane1/default).\n'
}

status_all_stacks() {
    printf '===== lane0 (GPU %s) =====\n' "$GPU_A"
    TTS_INSTANCE=lane0 \
    TTS_CONTROLLER_PORT="$LANE0_CONTROLLER_PORT" \
    TTS_POLICY_PORT="$LANE0_POLICY_PORT" \
    TTS_PRM_PORT="$LANE0_PRM_PORT" \
        bash "$RUNNER" status || true
    printf '===== lane1 (GPU %s) =====\n' "$GPU_B"
    TTS_INSTANCE=lane1 \
    TTS_CONTROLLER_PORT="$LANE1_CONTROLLER_PORT" \
    TTS_POLICY_PORT="$LANE1_POLICY_PORT" \
    TTS_PRM_PORT="$LANE1_PRM_PORT" \
        bash "$RUNNER" status || true
}

logs_all_stacks() {
    for instance_ports in \
        "lane0|$LANE0_CONTROLLER_PORT|$LANE0_POLICY_PORT|$LANE0_PRM_PORT" \
        "lane1|$LANE1_CONTROLLER_PORT|$LANE1_POLICY_PORT|$LANE1_PRM_PORT"
    do
        IFS='|' read -r instance cport pport rport <<<"$instance_ports"
        printf '\n######## instance=%s ########\n' "$instance"
        TTS_INSTANCE="$instance" \
        TTS_CONTROLLER_PORT="$cport" \
        TTS_POLICY_PORT="$pport" \
        TTS_PRM_PORT="$rport" \
            bash "$RUNNER" logs || true
    done
}

service_status_text() {
    local instance="$1" cport="$2" pport="$3" rport="$4"
    TTS_INSTANCE="$instance" \
    TTS_CONTROLLER_PORT="$cport" \
    TTS_POLICY_PORT="$pport" \
    TTS_PRM_PORT="$rport" \
        bash "$RUNNER" status 2>/dev/null || true
}

wait_for_models() {
    local instance="$1" cport="$2" pport="$3" rport="$4"
    local policy_model="$5" prm_model="$6"
    local attempt status
    for attempt in $(seq 1 90); do
        status="$(service_status_text "$instance" "$cport" "$pport" "$rport")"
        if grep -F "$policy_model" <<<"$status" >/dev/null &&
            grep -F "$prm_model" <<<"$status" >/dev/null; then
            printf 'Services ready for instance=%s after %s checks.\n' "$instance" "$attempt"
            return 0
        fi
        sleep 5
    done
    TTS_INSTANCE="$instance" \
    TTS_CONTROLLER_PORT="$cport" \
    TTS_POLICY_PORT="$pport" \
    TTS_PRM_PORT="$rport" \
        bash "$RUNNER" logs || true
    die "timed out waiting for instance=$instance models to register"
}

start_stack() {
    local instance="$1"
    local policy_gpu="$2"
    local prm_gpu="$3"
    local cport="$4"
    local pport="$5"
    local rport="$6"
    local policy_model="$7"
    local prm_model="$8"
    local mem_util="$9"

    stop_instance "$instance" "$cport" "$pport" "$rport"

    TTS_INSTANCE="$instance" \
    TTS_HOST_ADDR=127.0.0.1 \
    TTS_CONTROLLER_PORT="$cport" \
    TTS_POLICY_PORT="$pport" \
    TTS_PRM_PORT="$rport" \
    TTS_POLICY_MODEL="$policy_model" \
    TTS_PRM_MODEL="$prm_model" \
    TTS_POLICY_GPU="$policy_gpu" \
    TTS_PRM_GPU="$prm_gpu" \
    TTS_POLICY_GPU_MEMORY_UTILIZATION="$mem_util" \
    TTS_MAX_MODEL_LENGTH="$MAX_MODEL_LENGTH" \
    TTS_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    TTS_SEED="$SEED" \
    TTS_STRICT_DETERMINISM="$STRICT_DETERMINISM" \
    TTS_POLICY_MAX_CONCURRENCY="$POLICY_MAX_CONCURRENCY" \
    TTS_PRM_MAX_CONCURRENCY="$PRM_MAX_CONCURRENCY" \
    HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
        bash "$RUNNER" start-beam

    wait_for_models "$instance" "$cport" "$pport" "$rport" "$policy_model" "$prm_model"
}

memory_util_for() {
    local policy_gpu="$1" prm_gpu="$2" prm_class="$3"
    if [[ -n "$PAPER_POLICY_GPU_MEMORY_UTILIZATION" ]]; then
        printf '%s\n' "$PAPER_POLICY_GPU_MEMORY_UTILIZATION"
        return 0
    fi
    if [[ "$policy_gpu" == "$prm_gpu" ]]; then
        if [[ "$prm_class" == "small" ]]; then
            # Qwen/Llama policy + Skywork-1.5B PRM fits on one 32GB 5090, but
            # 0.68 is stable at width=2 but can still starve PRM forwards at
            # width=4/8 on Qwen.  0.55 leaves more transient room for the full
            # paper width sweep; stability is more important than throughput.
            printf '0.55\n'
        else
            # 7B/8B PRMs plus a 1B/1.5B/3B policy are tight on a 32GB 5090.
            # Keep the vLLM KV budget conservative so PRM forwards have room.
            printf '0.35\n'
        fi
    else
        printf '0.90\n'
    fi
}

run_one_config() {
    local instance="$1" cport="$2" pport="$3" rport="$4"
    local policy_gpu="$5" prm_gpu="$6" mem_util="$7"
    local policy_model="$8" prm_model="$9"
    local policy_label="${10}" prm_label="${11}" beam="${12}" width="${13}"

    local run_id trace_dir save_run_dir backup_suffix backup_dir
    run_id="$(run_id_for "$policy_label" "$prm_label" "$beam" "$width")"
    trace_dir="$TRACE_BASE_DIR/$run_id"
    save_run_dir="$SAVE_BASE_DIR/$run_id"

    if config_done "$policy_label" "$prm_label" "$beam" "$width"; then
        if [[ "$RESUME" == "1" ]]; then
            printf 'Skip existing: %s\n' "$run_id"
            return 0
        fi
        die "trace already exists: $trace_dir (set TTS_RESUME=1 to skip)"
    fi
    if [[ -e "$trace_dir" ]]; then
        if [[ "$RESUME" == "1" ]]; then
            backup_suffix="incomplete-$(date -u +%Y%m%d-%H%M%S)"
            backup_dir="${trace_dir}.${backup_suffix}"
            mv "$trace_dir" "$backup_dir"
            printf 'Moved incomplete trace aside: %s -> %s\n' "$trace_dir" "$backup_dir"
        else
            die "incomplete trace dir exists, choose a new TTS_RUN_STAMP: $trace_dir"
        fi
    fi
    if [[ -e "$save_run_dir" ]]; then
        if [[ "$RESUME" == "1" ]]; then
            backup_suffix="${backup_suffix:-incomplete-$(date -u +%Y%m%d-%H%M%S)}"
            backup_dir="${save_run_dir}.${backup_suffix}"
            mv "$save_run_dir" "$backup_dir"
            printf 'Moved incomplete result aside: %s -> %s\n' "$save_run_dir" "$backup_dir"
        else
            die "incomplete result dir exists, choose a new TTS_RUN_STAMP: $save_run_dir"
        fi
    fi

    printf '\n===== RUN %s  instance=%s gpu_policy=%s gpu_prm=%s =====\n' \
        "$run_id" "$instance" "$policy_gpu" "$prm_gpu"

    TTS_INSTANCE="$instance" \
    TTS_HOST_ADDR=127.0.0.1 \
    TTS_CONTROLLER_PORT="$cport" \
    TTS_POLICY_PORT="$pport" \
    TTS_PRM_PORT="$rport" \
    TTS_POLICY_MODEL="$policy_model" \
    TTS_PRM_MODEL="$prm_model" \
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
    TTS_TEMPERATURE="$TEMPERATURE" \
    TTS_SEED="$SEED" \
    TTS_STRICT_DETERMINISM="$STRICT_DETERMINISM" \
    TTS_VERIFY_DETERMINISM="$VERIFY_DETERMINISM" \
    TTS_POLICY_GPU="$policy_gpu" \
    TTS_PRM_GPU="$prm_gpu" \
    TTS_POLICY_GPU_MEMORY_UTILIZATION="$mem_util" \
    TTS_POLICY_MAX_CONCURRENCY="$POLICY_MAX_CONCURRENCY" \
    TTS_PRM_MAX_CONCURRENCY="$PRM_MAX_CONCURRENCY" \
    TTS_TRACE_BASE_DIR="$TRACE_BASE_DIR" \
    TTS_SAVE_BASE_DIR="$SAVE_BASE_DIR" \
    HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
        bash "$RUNNER" run-beam

    [[ -f "$trace_dir/manifest.json" ]] || die "run finished without manifest: $trace_dir"
    printf '===== DONE %s =====\n' "$run_id"
}

# Start services once for a model pair, then sweep all beams × widths.
run_pair_on_stack() {
    local instance="$1" policy_gpu="$2" prm_gpu="$3"
    local cport="$4" pport="$5" rport="$6"
    local policy_label="$7" prm_label="$8"

    local p_spec r_spec policy_model prm_model prm_class mem_util beam width widths
    p_spec="$(lookup_policy "$policy_label")"
    r_spec="$(lookup_prm "$prm_label")"
    IFS='|' read -r _ p_local <<<"$p_spec"
    IFS='|' read -r _ r_local prm_class <<<"$r_spec"
    policy_model="$(model_path "$p_local")"
    prm_model="$(model_path "$r_local")"
    mem_util="$(memory_util_for "$policy_gpu" "$prm_gpu" "$prm_class")"

    model_weights_ready "$policy_model" || die "policy weights missing: $policy_model"
    model_weights_ready "$prm_model" || die "PRM weights missing: $prm_model"

    if ! pair_needs_run "$policy_label" "$prm_label" && [[ "$RESUME" == "1" ]]; then
        printf 'All beam/width jobs done for %s|%s; skipping service start.\n' \
            "$policy_label" "$prm_label"
        return 0
    fi

    start_stack "$instance" "$policy_gpu" "$prm_gpu" "$cport" "$pport" "$rport" \
        "$policy_model" "$prm_model" "$mem_util"

    for beam in $BEAMS; do
        widths="$(widths_for_beam "$beam")"
        for width in $widths; do
            run_one_config "$instance" "$cport" "$pport" "$rport" \
                "$policy_gpu" "$prm_gpu" "$mem_util" \
                "$policy_model" "$prm_model" "$policy_label" "$prm_label" "$beam" "$width"
        done
    done

    stop_instance "$instance" "$cport" "$pport" "$rport"
}

run_lane_pairs() {
    local instance="$1" gpu="$2" cport="$3" pport="$4" rport="$5"
    shift 5
    local pair policy_label prm_label
    mkdir -p "$LOG_DIR"
    for pair in "$@"; do
        IFS='|' read -r policy_label prm_label <<<"$pair"
        printf 'Lane %s starting pair %s|%s on GPU %s\n' \
            "$instance" "$policy_label" "$prm_label" "$gpu" |
            tee -a "$LOG_DIR/${instance}.log"
        run_pair_on_stack "$instance" "$gpu" "$gpu" "$cport" "$pport" "$rport" \
            "$policy_label" "$prm_label" >>"$LOG_DIR/${instance}.log" 2>&1
    done
}

run_matrix() {
    validate_common
    check_models
    mkdir -p "$LOG_DIR" "$TRACE_BASE_DIR" "$SAVE_BASE_DIR"

    local -a lane0_jobs=() lane1_jobs=()
    local pair idx=0

    if ! mkdir "$RUN_LOCK_DIR" 2>/dev/null; then
        if [[ -f "$RUN_LOCK_DIR/pid" ]] && kill -0 "$(cat "$RUN_LOCK_DIR/pid")" 2>/dev/null; then
            die "paper trace run already active with pid $(cat "$RUN_LOCK_DIR/pid"); attach with: tmux attach -t orches-paper-traces"
        fi
        printf 'warning: removing stale paper trace run lock: %s\n' "$RUN_LOCK_DIR" >&2
        rm -f "$RUN_LOCK_DIR/pid"
        rmdir "$RUN_LOCK_DIR" 2>/dev/null || die "stale run lock is not empty: $RUN_LOCK_DIR"
        mkdir "$RUN_LOCK_DIR" || die "cannot create run lock: $RUN_LOCK_DIR"
    fi
    printf '%s\n' "$$" >"$RUN_LOCK_DIR/pid"
    trap 'rm -f "$RUN_LOCK_DIR/pid"; rmdir "$RUN_LOCK_DIR" 2>/dev/null || true' EXIT

    stop_all_stacks

    while IFS= read -r pair; do
        if (( idx % 2 == 0 )); then
            lane0_jobs+=("$pair")
        else
            lane1_jobs+=("$pair")
        fi
        idx=$((idx + 1))
    done < <(requested_pairs)

    printf 'Pair-sticky run: %s model pairs, beam(s)=%s, width(s)=%s, full %s unless overridden.\n' \
        "$idx" "$BEAMS" "$WIDTHS" "$TASK_LABEL"
    printf '  lane0 GPU%s pairs: %s\n' "$GPU_A" "${lane0_jobs[*]:-none}"
    printf '  lane1 GPU%s pairs: %s\n' "$GPU_B" "${lane1_jobs[*]:-none}"

    local -a pids=()
    if ((${#lane0_jobs[@]})); then
        run_lane_pairs lane0 "$GPU_A" "$LANE0_CONTROLLER_PORT" "$LANE0_POLICY_PORT" "$LANE0_PRM_PORT" \
            "${lane0_jobs[@]}" &
        pids+=($!)
    fi
    if ((${#lane1_jobs[@]})); then
        run_lane_pairs lane1 "$GPU_B" "$LANE1_CONTROLLER_PORT" "$LANE1_POLICY_PORT" "$LANE1_PRM_PORT" \
            "${lane1_jobs[@]}" &
        pids+=($!)
    fi

    local pid rc=0
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    (( rc == 0 )) || die "pair-sticky run failed; see $LOG_DIR/lane0.log and lane1.log"
    stop_instance lane0 "$LANE0_CONTROLLER_PORT" "$LANE0_POLICY_PORT" "$LANE0_PRM_PORT"
    stop_instance lane1 "$LANE1_CONTROLLER_PORT" "$LANE1_POLICY_PORT" "$LANE1_PRM_PORT"

    printf '\nPaper trace matrix finished. Stamp=%s\n' "$RUN_STAMP"
    print_plan
}

case "$COMMAND" in
    plan)
        validate_common
        print_plan
        ;;
    check)
        validate_common
        check_models
        ;;
    run)
        run_matrix
        ;;
    status)
        status_all_stacks
        ;;
    logs)
        logs_all_stacks
        ;;
    stop)
        stop_all_stacks
        ;;
    *)
        printf 'usage: %s {plan|check|run|status|logs|stop}\n' "${BASH_SOURCE[0]}" >&2
        exit 1
        ;;
esac
