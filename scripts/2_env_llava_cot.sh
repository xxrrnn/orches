#!/usr/bin/env bash
# Bootstrap the reproducible LLaVA-CoT inference environment, model, and data.
set -Eeuo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly SOURCE_DIR="$ROOT_DIR/third_party/LLaVA-CoT"
readonly ENV_DIR="$ROOT_DIR/environments/llava"
readonly PYTHON_BIN="${PYTHON_BIN:-python3.10}"
readonly MODELS_ROOT="${ORCHES_MODELS_ROOT:-$ROOT_DIR/models}"
readonly MODEL_REPO="${LLAVA_COT_MODEL:-Xkev/Llama-3.2V-11B-cot}"
readonly MODEL_PATH="${LLAVA_COT_MODEL_PATH:-$MODELS_ROOT/${MODEL_REPO##*/}}"
readonly DATASET_ROOT="${LLAVA_COT_DATASET_ROOT:-$MODELS_ROOT/LLaVA-CoT-100k}"
readonly DATASET_ID="${LLAVA_COT_DATASET:-Xkev/LLaVA-CoT-100k}"
readonly SOURCE_REVISION="081cc3fe3670fbff7b30fa22cbcafe031a2077bc"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

ensure_source() {
    if [[ ! -d "$SOURCE_DIR/.git" ]]; then
        mkdir -p "$(dirname "$SOURCE_DIR")"
        git clone https://github.com/PKU-YuanGroup/LLaVA-CoT.git "$SOURCE_DIR"
        git -C "$SOURCE_DIR" checkout --detach "$SOURCE_REVISION"
        return
    fi

    [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$SOURCE_REVISION" ]] ||
        die "$SOURCE_DIR is not at $SOURCE_REVISION; update it explicitly rather than overwriting it."
}

[[ -f "$ENV_DIR/pyproject.toml" ]] || die "missing LLaVA-CoT environment: $ENV_DIR"
ensure_source
mkdir -p "$MODELS_ROOT" "$DATASET_ROOT"

uv sync --directory "$ENV_DIR" --frozen --python "$PYTHON_BIN"

# LLaVA-CoT requires its processor fix, matching transformers==4.45.0 in the lockfile.
processor_path="$($ENV_DIR/.venv/bin/python -c 'import transformers.models.mllama.processing_mllama as module; print(module.__file__)')"
install -m 0644 "$SOURCE_DIR/inference/processing_mllama.py" "$processor_path"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    printf 'Downloading model %s into %s\n' "$MODEL_REPO" "$MODEL_PATH"
    bash "$ROOT_DIR/scripts/download_models.sh"
fi
[[ -f "$MODEL_PATH/config.json" ]] || die "missing model weights: $MODEL_PATH"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

printf 'Downloading dataset %s into %s\n' "$DATASET_ID" "$DATASET_ROOT"
LLAVA_COT_DATASET_ID="$DATASET_ID" LLAVA_COT_DATASET_ROOT="$DATASET_ROOT" "$ENV_DIR/.venv/bin/python" -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id=os.environ["LLAVA_COT_DATASET_ID"], repo_type="dataset", local_dir=os.path.join(os.environ["LLAVA_COT_DATASET_ROOT"], "dataset"))'

printf '\nLLaVA-CoT bootstrap complete.\n'
printf 'Environment: %s\nModel: %s\nDataset: %s\n' "$ENV_DIR" "$MODEL_PATH" "$DATASET_ROOT"
printf 'The 11B BF16 upstream demo needs a GPU with roughly 32 GiB; downloading does not require GPU memory.\n'
