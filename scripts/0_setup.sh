#!/usr/bin/env bash
# Fetch pinned sources and install a user-local CUDA Toolkit for Blackwell builds.
# Existing repositories and CUDA installations are only verified, never overwritten.
set -Eeuo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly CUDA_HOME="${ORCHES_CUDA_HOME:-$HOME/.local/cuda-12.8}"
readonly CUDA_RUNFILE_URL="${ORCHES_CUDA_RUNFILE_URL:-https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run}"
readonly CUDA_RUNFILE="$ROOT_DIR/.cache/cuda/$(basename "$CUDA_RUNFILE_URL")"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

download_file() {
    local url="$1"
    local destination="$2"
    mkdir -p "$(dirname "$destination")"
    if command -v curl >/dev/null; then
        curl --fail --location --retry 3 --output "$destination" "$url"
    elif command -v wget >/dev/null; then
        wget --output-document="$destination" "$url"
    else
        die "curl or wget is required to download $url"
    fi
}

ensure_cuda_toolkit() {
    if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
        printf 'Verified user-local CUDA Toolkit: %s\n' "$CUDA_HOME"
        return
    fi

    command -v gcc >/dev/null || die "gcc is required to build vLLM"
    command -v g++ >/dev/null || die "g++ is required to build vLLM"
    printf 'Installing CUDA Toolkit into %s (the NVIDIA driver is not modified)\n' "$CUDA_HOME"
    [[ -f "$CUDA_RUNFILE" ]] || download_file "$CUDA_RUNFILE_URL" "$CUDA_RUNFILE"
    sh "$CUDA_RUNFILE" --silent --toolkit --toolkitpath="$CUDA_HOME" --defaultroot="$CUDA_HOME" --no-opengl-libs --override
    [[ -x "$CUDA_HOME/bin/nvcc" ]] || die "CUDA Toolkit installation completed without nvcc"
}

ensure_repository() {
    local name="$1"
    local repository="$2"
    local revision="$3"
    local destination="$4"

    if [[ ! -e "$destination" ]]; then
        printf 'Cloning %s at %s\n' "$name" "$revision"
        mkdir -p "$(dirname "$destination")"
        git clone "$repository" "$destination"
        git -C "$destination" checkout --detach "$revision"
        return
    fi

    [[ -d "$destination/.git" ]] || {
        printf 'error: %s exists but is not a Git checkout: %s\n' "$name" "$destination" >&2
        exit 1
    }
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] || {
        printf 'error: %s is not at the pinned revision %s: %s\n' "$name" "$revision" "$(git -C "$destination" rev-parse HEAD)" >&2
        exit 1
    }
    printf 'Verified %s at %s\n' "$name" "$revision"
}

ensure_repository \
    'Compute-Optimal-TTS' \
    'https://github.com/RyanLiu112/compute-optimal-tts.git' \
    '0ee2578af1f8d6cac445c9c4c72780528bb94556' \
    "$ROOT_DIR/third_party/compute-optimal-tts"
ensure_repository \
    'LLaVA-CoT' \
    'https://github.com/PKU-YuanGroup/LLaVA-CoT.git' \
    '081cc3fe3670fbff7b30fa22cbcafe031a2077bc' \
    "$ROOT_DIR/third_party/LLaVA-CoT"
ensure_repository \
    'vLLM' \
    'https://github.com/vllm-project/vllm.git' \
    'b6553be1bc75f046b00046a4ad7576364d03c835' \
    "$ROOT_DIR/third_party/vllm"
ensure_cuda_toolkit

printf '\nPinned third-party sources and CUDA Toolkit are ready.\n'
