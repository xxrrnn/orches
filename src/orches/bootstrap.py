"""Reconstruct and verify the frozen third-party source environment."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the frozen environment.
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = PROJECT_ROOT / "third_party.lock"

TOP_LEVEL_REPOSITORIES = (
    "AttAcc simulator",
    "Duplex LLMSimulator",
    "Compute-optimal TTS",
    "LLaVA-o1",
)


class BootstrapError(RuntimeError):
    """Raised when the frozen environment cannot be reconstructed safely."""


@dataclass(frozen=True)
class RepositoryLock:
    """The checkout fields required from one third_party.lock entry."""

    name: str
    path: Path
    url: str
    commit: str


def _run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    """Run a visible command and fail immediately on a nonzero exit status."""

    location = cwd or PROJECT_ROOT
    print(f"+ ({location}) {shlex.join(command)}")
    subprocess.run(command, cwd=location, check=True)


def _output(command: Sequence[str], *, cwd: Path | None = None) -> str:
    """Run a query command and return its stripped standard output."""

    completed = subprocess.run(
        command,
        cwd=cwd or PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _is_git_checkout(path: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--git-dir"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _require_tools() -> None:
    missing = [tool for tool in ("git", "cmake", "ninja") if shutil.which(tool) is None]
    if missing:
        raise BootstrapError(f"missing required tools: {', '.join(missing)}")


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BootstrapError(f"{context} must be a TOML table")
    return value


def load_repository_locks(path: Path = LOCK_PATH) -> dict[str, RepositoryLock]:
    """Load repository definitions from the single version source of truth."""

    with path.open("rb") as lock_file:
        document = tomllib.load(lock_file)

    entries = document.get("repository")
    if not isinstance(entries, list):
        raise BootstrapError("third_party.lock must define [[repository]] entries")

    locks: dict[str, RepositoryLock] = {}
    for index, entry_value in enumerate(entries):
        entry = _as_mapping(entry_value, f"repository[{index}]")
        required = ("name", "path", "url", "commit")
        missing = [key for key in required if not isinstance(entry.get(key), str)]
        if missing:
            raise BootstrapError(
                f"repository[{index}] has missing or non-string fields: "
                f"{', '.join(missing)}"
            )

        name = entry["name"]
        if name in locks:
            raise BootstrapError(f"duplicate repository name in lock file: {name}")
        locks[name] = RepositoryLock(
            name=name,
            path=PROJECT_ROOT / entry["path"],
            url=entry["url"],
            commit=entry["commit"],
        )
    return locks


def _verify_head(repository: RepositoryLock) -> None:
    actual = _output(["git", "-C", str(repository.path), "rev-parse", "HEAD"])
    if actual != repository.commit:
        raise BootstrapError(
            f"{repository.name} is at {actual}, expected {repository.commit}; "
            "the bootstrap refuses to overwrite an existing checkout"
        )
    print(f"verified {repository.name}: {actual}")


def _ensure_top_level_checkout(
    repository: RepositoryLock,
    *,
    verify_only: bool,
) -> None:
    if not _is_git_checkout(repository.path):
        if verify_only:
            raise BootstrapError(f"missing checkout: {repository.path}")
        if repository.path.exists() and any(repository.path.iterdir()):
            raise BootstrapError(
                f"refusing to clone into non-empty directory: {repository.path}"
            )
        repository.path.parent.mkdir(parents=True, exist_ok=True)
        _run(
            ["git", "clone", "--no-checkout", repository.url, str(repository.path)]
        )
        _run(
            ["git", "-C", str(repository.path), "checkout", "--detach", repository.commit]
        )
    _verify_head(repository)


def _ensure_duplex_ramulator(
    locks: Mapping[str, RepositoryLock],
    *,
    verify_only: bool,
) -> None:
    duplex = locks["Duplex LLMSimulator"]
    ramulator = locks["Duplex Ramulator2"]
    if not _is_git_checkout(ramulator.path):
        if verify_only:
            raise BootstrapError(f"missing Duplex Ramulator2 submodule: {ramulator.path}")
        _run(
            ["git", "submodule", "update", "--init", "--recursive", "src/dram/ramulator2"],
            cwd=duplex.path,
        )
    _verify_head(ramulator)


def _ensure_attacc_ramulator(
    locks: Mapping[str, RepositoryLock],
    *,
    verify_only: bool,
) -> None:
    attacc = locks["AttAcc simulator"]
    gitlink = locks["AttAcc Ramulator2 gitlink"]
    build_base = locks["AttAcc Ramulator2 build base"]
    executable = build_base.path / "ramulator2"

    if executable.is_file() and os.access(executable, os.X_OK):
        _verify_head(build_base)
        print(f"verified AttAcc Ramulator2 executable: {executable}")
        return

    if verify_only:
        raise BootstrapError(f"missing AttAcc Ramulator2 executable: {executable}")

    if not _is_git_checkout(gitlink.path):
        _run(
            ["git", "submodule", "update", "--init", "--recursive", "ramulator2"],
            cwd=attacc.path,
        )
    _verify_head(gitlink)

    status = _output(["git", "-C", str(gitlink.path), "status", "--porcelain"])
    if status:
        raise BootstrapError(
            "AttAcc Ramulator2 has local changes but no built executable; "
            "refusing to run the upstream reset-and-patch script"
        )

    _run(["bash", "set_pim_ramulator.sh"], cwd=attacc.path)
    _verify_head(build_base)

    build_directory = build_base.path / "build"
    _run(
        [
            "cmake",
            "-S",
            str(build_base.path),
            "-B",
            str(build_directory),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    _run(
        [
            "cmake",
            "--build",
            str(build_directory),
            "--parallel",
            str(os.cpu_count() or 1),
        ]
    )

    built_executable = build_directory / "ramulator2"
    if not built_executable.is_file():
        raise BootstrapError(f"Ramulator2 build did not produce {built_executable}")
    shutil.copy2(built_executable, executable)
    print(f"installed AttAcc Ramulator2 executable: {executable}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct sources pinned by third_party.lock without overwriting checkouts."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check revisions and the AttAcc executable without changing the workspace",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require_tools()
    locks = load_repository_locks()

    for name in TOP_LEVEL_REPOSITORIES:
        _ensure_top_level_checkout(locks[name], verify_only=args.verify_only)
    _ensure_duplex_ramulator(locks, verify_only=args.verify_only)
    _ensure_attacc_ramulator(locks, verify_only=args.verify_only)

    mode = "verified" if args.verify_only else "ready"
    print(f"ORCHES third-party environment {mode}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BootstrapError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"bootstrap error: {error}") from error
