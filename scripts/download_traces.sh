#!/usr/bin/env bash
# Pull ORCHES traces from the GPU server onto this machine (client-side).
#
# Run on another computer that can SSH to the server.  Prefer rsync (resume /
# incremental); fall back to scp -r if rsync is missing.
#
# Usage:
#   bash scripts/download_traces.sh              # whole traces/
#   bash scripts/download_traces.sh list         # remote listing only
#   bash scripts/download_traces.sh Llama3.2-1B  # one subtree
#   bash scripts/download_traces.sh Llama3.2-1B/Qwen2.5-1.5B-PRM/MATH500
#
# Overrides:
#   ORCHES_TRACES_SSH=user@host
#   ORCHES_TRACES_REMOTE=/abs/path/on/server/traces
#   ORCHES_TRACES_LOCAL=/abs/or/rel/local/traces
#   ORCHES_TRACES_PORT=22
#   ORCHES_TRACES_DRY_RUN=1
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults for this ORCHES host (Titan5090).
SSH_URL="${ORCHES_TRACES_SSH:-${ORCHES_TTS_SERVER_SSH_URL:-rn_xu29@Titan5090}}"
REMOTE_TRACES="${ORCHES_TRACES_REMOTE:-/data/home/rn_xu29/Simulator/orches/traces}"
LOCAL_TRACES="${ORCHES_TRACES_LOCAL:-$ROOT_DIR/traces}"
SSH_PORT="${ORCHES_TRACES_PORT:-22}"
DRY_RUN="${ORCHES_TRACES_DRY_RUN:-0}"
TARGET="${1:-all}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

# Shared SSH options for scp/rsync/ssh.
ssh_opts=(
    -o "StrictHostKeyChecking=accept-new"
    -o "ServerAliveInterval=30"
    -o "ServerAliveCountMax=3"
)

run_ssh() {
    ssh -p "$SSH_PORT" "${ssh_opts[@]}" "$SSH_URL" "$@"
}

print_usage() {
    cat <<EOF
Usage: bash scripts/download_traces.sh [all|list|<relative-path-under-traces>]

Examples:
  bash scripts/download_traces.sh
  bash scripts/download_traces.sh list
  ORCHES_TRACES_SSH=rn_xu29@Titan5090 bash scripts/download_traces.sh Llama3.2-1B

Environment:
  ORCHES_TRACES_SSH      default: rn_xu29@Titan5090
  ORCHES_TRACES_REMOTE   default: /data/home/rn_xu29/Simulator/orches/traces
  ORCHES_TRACES_LOCAL    default: <repo>/traces
  ORCHES_TRACES_PORT     default: 22
  ORCHES_TRACES_DRY_RUN  1 = print only
EOF
}

# Normalize "traces/Foo" -> "Foo"; "all"/"" -> empty (whole tree).
normalize_rel() {
    local rel="$1"
    case "$rel" in
        all|.|"")
            printf '\n'
            ;;
        *)
            rel="${rel#traces/}"
            rel="${rel#/}"
            rel="${rel%/}"
            printf '%s\n' "$rel"
            ;;
    esac
}

list_remote() {
    printf 'Remote: %s:%s\n\n' "$SSH_URL" "$REMOTE_TRACES"
    run_ssh "ls -lah -- $(printf '%q' "$REMOTE_TRACES")"
    printf '\nDisk usage:\n'
    run_ssh "du -sh -- $(printf '%q' "$REMOTE_TRACES") $(printf '%q' "$REMOTE_TRACES")/* 2>/dev/null | sort -h"
}

ensure_remote_dir() {
    local path="$1"
    run_ssh "test -d $(printf '%q' "$path")" ||
        die "remote directory missing: $SSH_URL:$path"
}

download_rsync() {
    local remote_src="$1" # must end with / to copy contents
    local local_dest="$2"
    local -a opts=(
        -a
        --human-readable
        --compress
        --partial
        --info=stats2,progress2
        -e "ssh -p ${SSH_PORT} ${ssh_opts[*]}"
    )
    [[ "$DRY_RUN" == "1" ]] && opts+=(--dry-run)

    mkdir -p "$local_dest"
    printf 'rsync  %s:%s\n    -> %s/\n' "$SSH_URL" "$remote_src" "$local_dest"
    rsync "${opts[@]}" "${SSH_URL}:${remote_src}" "$local_dest/"
}

download_scp() {
    local remote_path="$1" # directory to copy
    local local_parent="$2"

    mkdir -p "$local_parent"
    printf 'scp -r %s:%s\n     -> %s/\n' "$SSH_URL" "$remote_path" "$local_parent"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'dry-run only\n'
        return 0
    fi
    scp -P "$SSH_PORT" -r "${ssh_opts[@]}" \
        "${SSH_URL}:${remote_path}" "$local_parent/"
}

main() {
    need_cmd ssh

    case "$TARGET" in
        -h|--help|help)
            print_usage
            exit 0
            ;;
        list)
            list_remote
            exit 0
            ;;
    esac

    local rel remote_path local_dest
    rel="$(normalize_rel "$TARGET")"

    if [[ -z "$rel" ]]; then
        remote_path="$REMOTE_TRACES"
        local_dest="$LOCAL_TRACES"
    else
        remote_path="$REMOTE_TRACES/$rel"
        local_dest="$LOCAL_TRACES/$rel"
    fi

    printf 'SSH    : %s (port %s)\n' "$SSH_URL" "$SSH_PORT"
    printf 'Remote : %s\n' "$remote_path"
    printf 'Local  : %s\n' "$local_dest"
    [[ "$DRY_RUN" == "1" ]] && printf 'Mode   : dry-run\n'

    ensure_remote_dir "$remote_path"

    if command -v rsync >/dev/null 2>&1; then
        # Trailing slash => copy directory contents into local_dest.
        download_rsync "${remote_path}/" "$local_dest"
    else
        need_cmd scp
        printf 'note: rsync not found; using scp -r (no resume)\n' >&2
        # scp copies the leaf dir name into the parent.
        download_scp "$remote_path" "$(dirname "$local_dest")"
    fi

    printf '\nDone.\n'
    if [[ -d "$local_dest" ]]; then
        du -sh "$local_dest" 2>/dev/null || true
    fi
}

main "$@"
