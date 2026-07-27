#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM_DIR=${ROOT_DIR}/third_party/compute-optimal-tts
PATCH_FILE=${ROOT_DIR}/integrations/compute-optimal-tts/0001-expose-exact-policy-tokens.patch
EXPECTED_REVISION=0ee2578af1f8d6cac445c9c4c72780528bb94556

actual_revision=$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)
if [[ ${actual_revision} != "${EXPECTED_REVISION}" ]]; then
    echo "compute-optimal-TTS revision mismatch: ${actual_revision}" >&2
    exit 2
fi

if [[ ${1:-} == "--check" ]]; then
    git -C "${UPSTREAM_DIR}" apply --check "${PATCH_FILE}"
    exit 0
fi

if [[ ${1:-} == "--reverse" ]]; then
    git -C "${UPSTREAM_DIR}" apply --check --reverse "${PATCH_FILE}"
    git -C "${UPSTREAM_DIR}" apply --reverse "${PATCH_FILE}"
    exit 0
fi

git -C "${UPSTREAM_DIR}" apply --check "${PATCH_FILE}"
git -C "${UPSTREAM_DIR}" apply "${PATCH_FILE}"
