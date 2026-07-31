#!/usr/bin/env bash
# Bootstrap the locked Compute-Optimal-TTS runtime and all paper-scale model pairs.
set -Eeuo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly SOURCE_DIR="$ROOT_DIR/third_party/compute-optimal-tts"
readonly TARGET="${TTS_TARGET:-rtx5090}"
readonly ENV_DIR="$ROOT_DIR/environments/compute-optimal-tts/$TARGET"
readonly PYTHON_BIN="${PYTHON_BIN:-python3.10}"
readonly HF_CACHE="${ORCHES_TTS_CACHE:-$HOME/.cache/huggingface/orches-tts}"
# 1.5B + 1.5B and 7B + 1.5B both use the Skywork verifier.
readonly QWEN_MATH_1_5B="Qwen/Qwen2.5-Math-1.5B-Instruct"
readonly QWEN_MATH_7B="Qwen/Qwen2.5-Math-7B-Instruct"
readonly SKYWORK_PRM_1_5B="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
# The verifier-heavy 1.5B + 7B setting.
readonly MATH_SHEPHERD_PRM_7B="peiyi9979/math-shepherd-mistral-7b-prm"
readonly SOURCE_REVISION="0ee2578af1f8d6cac445c9c4c72780528bb94556"
readonly PATCH_FILE="$ROOT_DIR/patches/compute-optimal-tts/001-tts-rtx50-compatibility.patch"
readonly VLLM_SOURCE_DIR="$ROOT_DIR/third_party/vllm"
readonly VLLM_REVISION="b6553be1bc75f046b00046a4ad7576364d03c835"
readonly CUDA_HOME="${ORCHES_CUDA_HOME:-$HOME/.local/cuda-12.8}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

download_model() {
    local model_id="$1"
    printf 'Downloading %s into %s\n' "$model_id" "$HF_HOME"
    TTS_MODEL_ID="$model_id" "$ENV_DIR/.venv/bin/python" -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id=os.environ["TTS_MODEL_ID"], cache_dir=os.environ["HF_HOME"])'
}

ensure_source() {
    if [[ ! -d "$SOURCE_DIR/.git" ]]; then
        mkdir -p "$(dirname "$SOURCE_DIR")"
        git clone https://github.com/RyanLiu112/compute-optimal-tts.git "$SOURCE_DIR"
        git -C "$SOURCE_DIR" checkout --detach "$SOURCE_REVISION"
    fi
    [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$SOURCE_REVISION" ]] ||
        die "$SOURCE_DIR is not at $SOURCE_REVISION; update it explicitly rather than overwriting it."
}

apply_compatibility_patch() {
    git -C "$SOURCE_DIR" apply --check "$PATCH_FILE" 2>/dev/null && {
        git -C "$SOURCE_DIR" apply "$PATCH_FILE"
        return
    }
    git -C "$SOURCE_DIR" apply --reverse --check "$PATCH_FILE" 2>/dev/null ||
        die "the TTS compatibility patch is neither cleanly applicable nor already applied."
}

[[ -f "$ENV_DIR/pyproject.toml" ]] || die "unknown TTS_TARGET '$TARGET'"
ensure_source
apply_compatibility_patch

if [[ "$TARGET" == "rtx5090" ]]; then
    [[ -x "$CUDA_HOME/bin/nvcc" ]] || die "run scripts/0_setup.sh to install the user-local CUDA Toolkit"
    [[ -d "$VLLM_SOURCE_DIR/.git" ]] || die "run scripts/0_setup.sh to fetch the pinned vLLM source"
    [[ "$(git -C "$VLLM_SOURCE_DIR" rev-parse HEAD)" == "$VLLM_REVISION" ]] ||
        die "$VLLM_SOURCE_DIR is not at the pinned vLLM revision"
    export CUDA_HOME
    export CUDACXX="$CUDA_HOME/bin/nvcc"
    export PATH="$CUDA_HOME/bin:$PATH"
    export VLLM_TARGET_DEVICE=cuda
    export TORCH_CUDA_ARCH_LIST=12.0
    export CMAKE_CUDA_ARCHITECTURES=120
    export MAX_JOBS="${VLLM_MAX_JOBS:-4}"
    # vLLM treats any nonempty value, including "0", as precompiled mode.
    unset VLLM_USE_PRECOMPILED
fi

mkdir -p "$HF_CACHE"
if [[ "$TARGET" == "rtx5090" ]]; then
    # The lockfile points to the pinned source tree.  Build it in the final UV
    # environment, using the CUDA 12.8 toolkit installed by step 0.
    uv sync --directory "$ENV_DIR" --frozen --no-build-isolation --python "$PYTHON_BIN"
    vllm_fa2="$($ENV_DIR/.venv/bin/python -c 'import vllm.vllm_flash_attn._vllm_fa2_C as module; print(module.__file__)')"
    "$CUDA_HOME/bin/cuobjdump" --list-elf "$vllm_fa2" | grep -q 'sm_120' ||
        die "the vLLM FlashAttention extension does not contain sm_120"
else
    uv sync --directory "$ENV_DIR" --frozen --python "$PYTHON_BIN"
fi

# hf-mirror.com is incompatible with huggingface_hub metadata redirects in this setup.
export HF_HOME="$HF_CACHE"
export HF_ENDPOINT="https://huggingface.co"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

# Download the four unique weights needed for the three requested configurations:
#   1.5B + 1.5B: Qwen2.5-Math-1.5B + Skywork-o1-PRM-1.5B
#   1.5B + 7B:   Qwen2.5-Math-1.5B + Math-Shepherd-Mistral-7B-PRM
#   7B + 1.5B:   Qwen2.5-Math-7B   + Skywork-o1-PRM-1.5B
download_model "$QWEN_MATH_1_5B"
download_model "$QWEN_MATH_7B"
download_model "$SKYWORK_PRM_1_5B"
download_model "$MATH_SHEPHERD_PRM_7B"

# The AIME24, AMC23, and MATH-500 JSONL assets used by the upstream evaluator
# are versioned with the pinned source tree; fail early if the smoke dataset is absent.
readonly AIME24_DATASET="$SOURCE_DIR/src/envs/MATH/dataset/test_aime.jsonl"
[[ -s "$AIME24_DATASET" ]] || die "missing bundled AIME24 smoke dataset: $AIME24_DATASET"

printf '\nCompute-Optimal-TTS bootstrap complete.\n'
printf 'Environment: %s\nCache: %s\nAIME24: %s\n' "$ENV_DIR" "$HF_HOME" "$AIME24_DATASET"
