#!/usr/bin/env bash
# Download model weights into flat trees under models/<short-name>/.
#
# Usage:
#   bash scripts/download_models.sh
set -Eeuo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ENV_DIR="$ROOT_DIR/environments/tts"
readonly MODELS_ROOT="${ORCHES_MODELS_ROOT:-$ROOT_DIR/models}"
readonly HF_BIN="${ORCHES_HF_BIN:-$ENV_DIR/.venv/bin/hf}"
readonly LEGACY_HUB_CACHE="${ORCHES_LEGACY_HUB_CACHE:-$MODELS_ROOT/hub}"
readonly STATUS_FILE="$MODELS_ROOT/download-models.status"
readonly HF_ENDPOINT_MIRROR="${HF_ENDPOINT_MIRROR:-https://hf-mirror.com}"
readonly HF_ENDPOINT_OFFICIAL="${HF_ENDPOINT_OFFICIAL:-https://huggingface.co}"
readonly HF_FALLBACK_PROXY="${HF_FALLBACK_PROXY:-http://127.0.0.1:7890}"
readonly CONTINUE_ON_ERROR="${ORCHES_DOWNLOAD_CONTINUE:-0}"

readonly -a MODEL_REPOS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
    "RLHFlow/Llama3.1-8B-PRM-Mistral-Data"
    "Xkev/Llama-3.2V-11B-cot"
)

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

if [[ ! -x "$HF_BIN" ]]; then
    printf 'Bootstrapping TTS environment (uv sync)...\n'
    uv sync --directory "$ENV_DIR" --frozen --python "$(uv python find --managed-python 3.10)"
fi

cd "$ROOT_DIR"
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

[[ -x "$HF_BIN" ]] || die "missing hf CLI: $HF_BIN (run: bash scripts/1_env_compute_optimal_tts.sh)"

with_proxy() {
    export https_proxy="$HF_FALLBACK_PROXY"
    export http_proxy="$HF_FALLBACK_PROXY"
    export HTTPS_PROXY="$HF_FALLBACK_PROXY"
    export HTTP_PROXY="$HF_FALLBACK_PROXY"
}

without_proxy() {
    unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY all_proxy
}

local_dir_for_repo() {
    printf '%s/%s\n' "$MODELS_ROOT" "${1##*/}"
}

model_is_complete() {
    local dir="$1"
    python3 - "$dir" <<'PY'
import json
import os
import sys

MIN_BYTES = 1_000_000


def ok_file(path: str) -> bool:
    try:
        resolved = os.path.realpath(path)
        return os.path.isfile(resolved) and os.path.getsize(resolved) > MIN_BYTES
    except OSError:
        return False


def check_index(model_dir: str, index_name: str) -> bool:
    index_path = os.path.join(model_dir, index_name)
    if not os.path.isfile(index_path):
        return False
    with open(index_path, encoding="utf-8") as handle:
        data = json.load(handle)
    shards = sorted(set(data.get("weight_map", {}).values()))
    if not shards:
        return False
    return all(ok_file(os.path.join(model_dir, shard)) for shard in shards)


model_dir = sys.argv[1]
if not os.path.isfile(os.path.join(model_dir, "config.json")):
    sys.exit(1)

if os.path.lexists(os.path.join(model_dir, "model.safetensors")):
    sys.exit(0 if ok_file(os.path.join(model_dir, "model.safetensors")) else 1)
if os.path.isfile(os.path.join(model_dir, "model.safetensors.index.json")):
    sys.exit(0 if check_index(model_dir, "model.safetensors.index.json") else 1)
if os.path.lexists(os.path.join(model_dir, "pytorch_model.bin")):
    sys.exit(0 if ok_file(os.path.join(model_dir, "pytorch_model.bin")) else 1)
if os.path.isfile(os.path.join(model_dir, "pytorch_model.bin.index.json")):
    sys.exit(0 if check_index(model_dir, "pytorch_model.bin.index.json") else 1)

sys.exit(1)
PY
}

legacy_hub_cache_dir() {
    local repo="$1"
    printf '%s/models--%s' "$LEGACY_HUB_CACHE" "$(echo "$repo" | sed 's|/|--|g')"
}

local_dir_has_broken_hub_symlinks() {
    local dir="$1" link target

    while IFS= read -r -d '' link; do
        target="$(readlink "$link")"
        [[ "$target" == ../* ]] && return 0
    done < <(find "$dir" -type l -print0 2>/dev/null)
    return 1
}

legacy_hub_has_weights() {
    local repo="$1"
    local hub_dir snap ref

    hub_dir="$(legacy_hub_cache_dir "$repo")"
    [[ -f "$hub_dir/refs/main" ]] || return 1
    ref="$(<"$hub_dir/refs/main")"
    snap="$hub_dir/snapshots/$ref"
    [[ -d "$snap" ]] || return 1
    model_is_complete "$snap"
}

migrate_legacy_hub() {
    local repo="$1" local_dir="$2"
    local hub_dir snap ref

    hub_dir="$(legacy_hub_cache_dir "$repo")"
    [[ -f "$hub_dir/refs/main" ]] || return 1
    ref="$(<"$hub_dir/refs/main")"
    snap="$hub_dir/snapshots/$ref"
    [[ -d "$snap" ]] || return 1
    legacy_hub_has_weights "$repo" || return 1

    printf 'MIGRATE %s legacy hub -> %s\n' "$repo" "$local_dir" | tee -a "$STATUS_FILE"
    rm -rf "$local_dir"
    mkdir -p "$local_dir"
    rsync -aL "$snap/" "$local_dir/"
    model_is_complete "$local_dir"
}

run_download() {
    local endpoint="$1" repo="$2" local_dir="$3"
    HF_ENDPOINT="$endpoint" "$HF_BIN" download "$repo" \
        --local-dir "$local_dir" \
        --exclude "original/*" \
        --max-workers 1
}

try_download_endpoint() {
    local endpoint="$1" repo="$2" local_dir="$3" proxy_mode="$4"
    local proxy_label="off"

    if [[ "$proxy_mode" == with-proxy ]]; then
        proxy_label="$HF_FALLBACK_PROXY"
        with_proxy
    else
        without_proxy
    fi

    printf 'TRY %s endpoint=%s proxy=%s\n' "$repo" "$endpoint" "$proxy_label" | tee -a "$STATUS_FILE"
    if run_download "$endpoint" "$repo" "$local_dir" && model_is_complete "$local_dir"; then
        printf 'OK %s endpoint=%s proxy=%s\n' "$repo" "$endpoint" "$proxy_label" | tee -a "$STATUS_FILE"
        [[ "$proxy_mode" == with-proxy ]] && without_proxy
        return 0
    fi
    [[ "$proxy_mode" == with-proxy ]] && without_proxy
    printf 'FAIL %s endpoint=%s proxy=%s\n' "$repo" "$endpoint" "$proxy_label" | tee -a "$STATUS_FILE"
    return 1
}

download_repo() {
    local repo="$1"
    local local_dir

    local_dir="$(local_dir_for_repo "$repo")"
    mkdir -p "$local_dir"

    printf 'START %s dir=%s\n' "$repo" "$local_dir" | tee -a "$STATUS_FILE"

    if local_dir_has_broken_hub_symlinks "$local_dir" || ! model_is_complete "$local_dir"; then
        migrate_legacy_hub "$repo" "$local_dir" || true
    fi
    if model_is_complete "$local_dir"; then
        printf 'SKIP %s already complete in %s\n' "$repo" "$local_dir" | tee -a "$STATUS_FILE"
        return 0
    fi

    if try_download_endpoint "$HF_ENDPOINT_MIRROR" "$repo" "$local_dir" without-proxy; then
        return 0
    fi
    if try_download_endpoint "$HF_ENDPOINT_OFFICIAL" "$repo" "$local_dir" with-proxy; then
        return 0
    fi
    return 1
}

mkdir -p "$MODELS_ROOT"
printf '\n=== download run %s ===\n' "$(date -Iseconds)" >> "$STATUS_FILE"

failures=0
for repo in "${MODEL_REPOS[@]}"; do
    [[ -n "$repo" ]] || continue
    if ! download_repo "$repo"; then
        failures=$((failures + 1))
        [[ "$CONTINUE_ON_ERROR" == 1 ]] || die "download failed: $repo"
    fi
done

without_proxy

if (( failures > 0 )); then
    die "$failures model download(s) failed; see $STATUS_FILE"
fi

printf 'All %s model download(s) complete under %s\n' "${#MODEL_REPOS[@]}" "$MODELS_ROOT"
