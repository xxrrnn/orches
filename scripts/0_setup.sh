#!/usr/bin/env bash
# Fetch the pinned upstream repositories required by the ORCHES environments.
# Existing repositories are only verified, never reset or overwritten.
set -Eeuo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

printf '\nPinned third-party sources are ready.\n'
