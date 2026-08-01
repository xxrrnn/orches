#!/usr/bin/env bash
# Bootstrap the locked Compute-Optimal-TTS runtime and all paper-scale model pairs.
set -Eeuo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly SOURCE_DIR="$ROOT_DIR/third_party/compute-optimal-tts"
readonly ENV_DIR="$ROOT_DIR/environments/tts"
readonly PYTHON_BIN="${PYTHON_BIN:-$(uv python find --managed-python 3.10)}"
readonly HF_CACHE="${ORCHES_TTS_CACHE:-$ROOT_DIR/models}"
readonly HF_BIN="$ENV_DIR/.venv/bin/hf"
readonly VLLM_WHEEL_URL="${ORCHES_VLLM_WHEEL_URL:-https://github.com/xxrrnn/orches/releases/download/whl/vllm-0.9.1-cp310-cp310-linux_x86_64.whl}"
readonly VLLM_WHEEL="$ROOT_DIR/artifacts/vllm-sm120/vllm-0.9.1-cp310-cp310-linux_x86_64.whl"
readonly VLLM_WHEEL_SHA256="937bd9dbfaadaf816c2857c7ae0e1b73bfe805a0c43af030ceb034f4132da74a"
# 1.5B + 1.5B and 7B + 1.5B both use the Skywork verifier.
readonly QWEN_MATH_1_5B="Qwen/Qwen2.5-Math-1.5B-Instruct"
readonly QWEN_MATH_7B="Qwen/Qwen2.5-Math-7B-Instruct"
readonly QWEN_0_5B="Qwen/Qwen2.5-0.5B-Instruct"
readonly SKYWORK_PRM_1_5B="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
# The verifier-heavy 1.5B + 7B setting.
readonly MATH_SHEPHERD_PRM_7B="peiyi9979/math-shepherd-mistral-7b-prm"
readonly SOURCE_REVISION="0ee2578af1f8d6cac445c9c4c72780528bb94556"
readonly PATCH_FILES=(
    "$ROOT_DIR/patches/compute-optimal-tts/001-tts-rtx50-compatibility.patch"
    "$ROOT_DIR/patches/compute-optimal-tts/002-tts-baseline-tracing.patch"
    "$ROOT_DIR/patches/compute-optimal-tts/003-vllm-multisample-aggregation.patch"
    "$ROOT_DIR/patches/compute-optimal-tts/004-compact-output-token-text.patch"
    "$ROOT_DIR/patches/compute-optimal-tts/005-reward-score-alias.patch"
    "$ROOT_DIR/patches/compute-optimal-tts/006-software-candidate-token-lengths.patch"
)

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# Step 1: verify the selected locked environment exists.
[[ -f "$ENV_DIR/pyproject.toml" ]] || die "missing TTS environment: $ENV_DIR"

# Step 2: fetch or verify the pinned Compute-Optimal-TTS source checkout.
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    mkdir -p "$(dirname "$SOURCE_DIR")"
    git clone https://github.com/RyanLiu112/compute-optimal-tts.git "$SOURCE_DIR"
    git -C "$SOURCE_DIR" checkout --detach "$SOURCE_REVISION"
fi
[[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$SOURCE_REVISION" ]] ||
    die "$SOURCE_DIR is not at $SOURCE_REVISION; update it explicitly rather than overwriting it."

# Step 3: apply each local patch once. Later patches intentionally touch some
# of the same hunks, so reverse-applying an earlier patch is not a reliable
# installed-state test.  Use a small, unique marker for each patch instead.
patch_is_applied() {
    case "$(basename "$1")" in
        001-tts-rtx50-compatibility.patch)
            [[ -f "$SOURCE_DIR/src/envs/MATH/latex2sympy/pyproject.toml" ]] &&
                rg -q '_node_ip_address=local_ray_ip' "$SOURCE_DIR/src/reason/evaluation/evaluate.py"
            ;;
        002-tts-baseline-tracing.patch)
            [[ -f "$SOURCE_DIR/src/reason/tracing/tts_trace.py" ]] &&
                rg -q 'TRACE_SCHEMA_VERSION = "tts-trace-v1"' "$SOURCE_DIR/src/reason/tracing/tts_trace.py"
            ;;
        003-vllm-multisample-aggregation.patch)
            rg -q 'collected_outputs\[output.index\]' "$SOURCE_DIR/src/reason/llm_service/workers/vllm_worker.py"
            ;;
        004-compact-output-token-text.patch)
            rg -q 'Pretty-print JSON while keeping decoded token text on one line' "$SOURCE_DIR/src/reason/tracing/tts_trace.py"
            ;;
        005-reward-score-alias.patch)
            rg -q '"reward_score": float\(score\[-1\]\)' "$SOURCE_DIR/src/reason/tracing/tts_trace.py"
            ;;
        006-software-candidate-token-lengths.patch)
            rg -q '"token_length": int\(action.get\("num_token", 0\)\)' "$SOURCE_DIR/src/reason/tracing/tts_trace.py"
            ;;
        *) die "missing applied-state marker for patch: $1" ;;
    esac
}

for PATCH_FILE in "${PATCH_FILES[@]}"; do
    [[ -f "$PATCH_FILE" ]] || die "missing TTS patch: $PATCH_FILE"
    if patch_is_applied "$PATCH_FILE"; then
        continue
    fi
    git -C "$SOURCE_DIR" apply --whitespace=nowarn --check "$PATCH_FILE" 2>/dev/null ||
        die "the TTS patch is neither cleanly applicable nor recognized as applied: $PATCH_FILE"
    git -C "$SOURCE_DIR" apply --whitespace=nowarn "$PATCH_FILE"
done

# Step 4: this Blackwell environment uses the pinned local vLLM wheel.
mkdir -p "$(dirname "$VLLM_WHEEL")"
if [[ ! -f "$VLLM_WHEEL" ]]; then
    printf 'Downloading vLLM wheel:\n  %s\n  -> %s\n' "$VLLM_WHEEL_URL" "$VLLM_WHEEL"
    if command -v curl >/dev/null; then
        curl --fail --location --retry 3 --output "$VLLM_WHEEL" "$VLLM_WHEEL_URL"
    elif command -v wget >/dev/null; then
        wget --output-document="$VLLM_WHEEL" "$VLLM_WHEEL_URL"
    else
        die "curl or wget is required to download $VLLM_WHEEL_URL"
    fi
fi
[[ "$(sha256sum "$VLLM_WHEEL" | awk '{print $1}')" == "$VLLM_WHEEL_SHA256" ]] ||
    die "unexpected SHA256 for $VLLM_WHEEL"

# Step 5: install the locked Python environment.
mkdir -p "$HF_CACHE"
uv sync --directory "$ENV_DIR" --frozen --python "$PYTHON_BIN"

# Step 6: download the exact model weights used by the requested paper-scale
# configurations. Use hf-mirror by default, but allow overriding:
#   HF_ENDPOINT=https://huggingface.co bash scripts/1_env_compute_optimal_tts.sh
export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

# Download the local models used by the example and paper-scale configurations:
#   0.5B + 1.5B: Qwen2.5-0.5B      + Skywork-o1-PRM-1.5B
#   1.5B + 1.5B: Qwen2.5-Math-1.5B + Skywork-o1-PRM-1.5B
#   1.5B + 7B:   Qwen2.5-Math-1.5B + Math-Shepherd-Mistral-7B-PRM
#   7B + 1.5B:   Qwen2.5-Math-7B   + Skywork-o1-PRM-1.5B
"$HF_BIN" download "$QWEN_0_5B" --cache-dir "$HF_HUB_CACHE"
"$HF_BIN" download "$QWEN_MATH_1_5B" --cache-dir "$HF_HUB_CACHE"
"$HF_BIN" download "$QWEN_MATH_7B" --cache-dir "$HF_HUB_CACHE"
"$HF_BIN" download "$SKYWORK_PRM_1_5B" --cache-dir "$HF_HUB_CACHE"
"$HF_BIN" download "$MATH_SHEPHERD_PRM_7B" --cache-dir "$HF_HUB_CACHE"

# Step 7: verify the bundled smoke-test dataset is present.
# The AIME24, AMC23, and MATH-500 JSONL assets used by the upstream evaluator
# are versioned with the pinned source tree; fail early if the smoke dataset is absent.
readonly AIME24_DATASET="$SOURCE_DIR/src/envs/MATH/dataset/test_aime.jsonl"
[[ -s "$AIME24_DATASET" ]] || die "missing bundled AIME24 smoke dataset: $AIME24_DATASET"

printf '\nCompute-Optimal-TTS bootstrap complete.\n'
printf 'Environment: %s\nHF_HOME: %s\nHF_HUB_CACHE: %s\nAIME24: %s\n' "$ENV_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$AIME24_DATASET"
