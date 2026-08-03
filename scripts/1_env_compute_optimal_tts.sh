#!/usr/bin/env bash
# Bootstrap the locked Compute-Optimal-TTS runtime and all paper-scale model pairs.
set -Eeuo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly SOURCE_DIR="$ROOT_DIR/third_party/compute-optimal-tts"
readonly ENV_DIR="$ROOT_DIR/environments/tts"
readonly PYTHON_BIN="${PYTHON_BIN:-$(uv python find --managed-python 3.10)}"
readonly HF_CACHE="${ORCHES_TTS_CACHE:-$ROOT_DIR/models}"
readonly VLLM_WHEEL_URL="${ORCHES_VLLM_WHEEL_URL:-https://github.com/xxrrnn/orches/releases/download/whl/vllm-0.9.1-cp310-cp310-linux_x86_64.whl}"
readonly VLLM_WHEEL="$ROOT_DIR/artifacts/vllm-sm120/vllm-0.9.1-cp310-cp310-linux_x86_64.whl"
readonly VLLM_WHEEL_SHA256="937bd9dbfaadaf816c2857c7ae0e1b73bfe805a0c43af030ceb034f4132da74a"
readonly SOURCE_REVISION="0ee2578af1f8d6cac445c9c4c72780528bb94556"
readonly PATCH_FILES=(
    "$ROOT_DIR/patches/compute-optimal-tts/001-tts-rtx50-compatibility.patch"
    "$ROOT_DIR/patches/compute-optimal-tts/002-tts-baseline-tracing.patch"
    "$ROOT_DIR/patches/compute-optimal-tts/003-vllm-multisample-aggregation.patch"
    "$ROOT_DIR/patches/compute-optimal-tts/004-compact-output-token-text.patch"
    "$ROOT_DIR/patches/compute-optimal-tts/005-reward-score-alias.patch"
    "$ROOT_DIR/patches/compute-optimal-tts/006-software-candidate-token-lengths.patch"
    "$ROOT_DIR/patches/compute-optimal-tts/007-tts-strict-determinism.patch"
    "$ROOT_DIR/patches/compute-optimal-tts/008-tts-prm-controller-routing.patch"
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
        007-tts-strict-determinism.patch)
            [[ -f "$SOURCE_DIR/src/reason/llm_service/workers/determinism.py" ]] &&
                rg -q 'strict determinism requires a request seed' "$SOURCE_DIR/src/reason/llm_service/workers/vllm_worker.py" &&
                rg -q 'attn_implementation' "$SOURCE_DIR/src/reason/llm_service/workers/reward_model_worker.py" &&
                rg -q 'seed=args.seed' "$SOURCE_DIR/src/reason/evaluation/evaluate.py"
            ;;
        008-tts-prm-controller-routing.patch)
            rg -q 'multi_gpu=self.multi_gpu, timeout=timeout' "$SOURCE_DIR/src/reason/inference/rm_call.py"
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
download_vllm_wheel() {
    printf 'Downloading vLLM wheel:\n  %s\n  -> %s\n' "$VLLM_WHEEL_URL" "$VLLM_WHEEL"
    if command -v curl >/dev/null; then
        curl --fail --location --retry 3 --output "$VLLM_WHEEL" "$VLLM_WHEEL_URL"
    elif command -v wget >/dev/null; then
        wget --output-document="$VLLM_WHEEL" "$VLLM_WHEEL_URL"
    else
        die "curl or wget is required to download $VLLM_WHEEL_URL"
    fi
}

verify_vllm_wheel() {
    local actual_sha256
    actual_sha256="$(sha256sum "$VLLM_WHEEL" | awk '{print $1}')"
    [[ "$actual_sha256" == "$VLLM_WHEEL_SHA256" ]] || {
        printf 'warning: unexpected SHA256 for %s\n' "$VLLM_WHEEL" >&2
        printf '  expected: %s\n' "$VLLM_WHEEL_SHA256" >&2
        printf '  actual:   %s\n' "$actual_sha256" >&2
        return 1
    }
    unzip -tq "$VLLM_WHEEL" >/dev/null 2>&1 || {
        printf 'warning: %s is not a valid wheel archive\n' "$VLLM_WHEEL" >&2
        return 1
    }
}

mkdir -p "$(dirname "$VLLM_WHEEL")"
if [[ ! -f "$VLLM_WHEEL" ]] || ! verify_vllm_wheel; then
    [[ -f "$VLLM_WHEEL" ]] && rm -f "$VLLM_WHEEL"
    download_vllm_wheel
fi
verify_vllm_wheel || die "vLLM wheel download failed verification for $VLLM_WHEEL"

# Step 5: install the locked Python environment.
mkdir -p "$HF_CACHE"
uv sync --directory "$ENV_DIR" --frozen --python "$PYTHON_BIN"

# Step 6: download model weights into models/<short-name>/ trees.
bash "$ROOT_DIR/scripts/download_models.sh"

# Step 7: verify the bundled smoke-test dataset is present.
# The AIME24, AMC23, and MATH-500 JSONL assets used by the upstream evaluator
# are versioned with the pinned source tree; fail early if the smoke dataset is absent.
readonly AIME24_DATASET="$SOURCE_DIR/src/envs/MATH/dataset/test_aime.jsonl"
[[ -s "$AIME24_DATASET" ]] || die "missing bundled AIME24 smoke dataset: $AIME24_DATASET"

printf '\nCompute-Optimal-TTS bootstrap complete.\n'
printf 'Environment: %s\nModels: %s\nAIME24: %s\n' "$ENV_DIR" "$HF_CACHE" "$AIME24_DATASET"
