"""Command-line entry point for the ORCHES third-party bootstrap."""

from __future__ import annotations

import subprocess

from orches.bootstrap import BootstrapError, main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BootstrapError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"bootstrap error: {error}") from error
