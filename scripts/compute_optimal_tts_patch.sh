#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM_DIR=${ROOT_DIR}/third_party/compute-optimal-tts
PATCH_FILES=(
    "${ROOT_DIR}/integrations/compute-optimal-tts/0001-expose-exact-policy-tokens.patch"
    "${ROOT_DIR}/integrations/compute-optimal-tts/0002-record-policy-control-flow.patch"
)
EXPECTED_REVISION=0ee2578af1f8d6cac445c9c4c72780528bb94556

actual_revision=$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)
if [[ ${actual_revision} != "${EXPECTED_REVISION}" ]]; then
    echo "compute-optimal-TTS revision mismatch: ${actual_revision}" >&2
    exit 2
fi

if [[ ${1:-} == "--check" ]]; then
    git -C "${UPSTREAM_DIR}" apply --check "${PATCH_FILES[@]}"
    exit 0
fi

if [[ ${1:-} == "--reverse" ]]; then
    REVERSED_PATCH_FILES=()
    for ((i=${#PATCH_FILES[@]} - 1; i >= 0; i--)); do
        REVERSED_PATCH_FILES+=("${PATCH_FILES[i]}")
    done
    git -C "${UPSTREAM_DIR}" apply --check --reverse "${REVERSED_PATCH_FILES[@]}"
    git -C "${UPSTREAM_DIR}" apply --reverse "${REVERSED_PATCH_FILES[@]}"
    exit 0
fi

git -C "${UPSTREAM_DIR}" apply --check "${PATCH_FILES[@]}"
git -C "${UPSTREAM_DIR}" apply "${PATCH_FILES[@]}"
