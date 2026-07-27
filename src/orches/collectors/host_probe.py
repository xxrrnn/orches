"""Trace-host runtime probing without importing GPU packages at module load."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import WorkloadTraceError
from ..workload.policy_manifest import PolicyRuntimeEnvironment


TRACE_HOST_PROBE_SCHEMA_VERSION = 1
_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class TraceHostProbe:
    """Runtime identity captured by the same UV interpreter as collection."""

    probe_schema_version: int
    runtime: PolicyRuntimeEnvironment
    python_executable: str
    driver_accelerators: tuple[str, ...]
    bf16_supported: bool
    collection_ready: bool
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.probe_schema_version != TRACE_HOST_PROBE_SCHEMA_VERSION:
            raise WorkloadTraceError(
                "unsupported trace host probe schema version "
                f"{self.probe_schema_version!r}"
            )
        if not self.python_executable.strip():
            raise WorkloadTraceError(
                "trace host probe python_executable must not be empty"
            )
        if not isinstance(self.bf16_supported, bool) or not isinstance(
            self.collection_ready, bool
        ):
            raise WorkloadTraceError(
                "trace host probe readiness fields must be booleans"
            )
        if any(not name.strip() for name in self.driver_accelerators):
            raise WorkloadTraceError(
                "trace host probe driver accelerator names must not be empty"
            )
        if self.bf16_supported and self.runtime.accelerator_count == 0:
            raise WorkloadTraceError(
                "trace host probe cannot support BF16 without an accelerator"
            )
        if len(set(self.warnings)) != len(self.warnings):
            raise WorkloadTraceError(
                "trace host probe warnings must not contain duplicates"
            )
        if any(not warning.strip() for warning in self.warnings):
            raise WorkloadTraceError(
                "trace host probe warnings must not contain empty strings"
            )
        expected_ready = (
            self.runtime.accelerator_count > 0
            and self.runtime.torch_version != "not-installed"
            and self.runtime.cuda_version != _UNAVAILABLE
            and self.runtime.driver_version != _UNAVAILABLE
        )
        if self.collection_ready is not expected_ready:
            raise WorkloadTraceError(
                "trace host probe collection_ready disagrees with runtime fields"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_schema_version": self.probe_schema_version,
            "runtime": self.runtime.to_dict(),
            "python_executable": self.python_executable,
            "driver_accelerators": list(self.driver_accelerators),
            "bf16_supported": self.bf16_supported,
            "collection_ready": self.collection_ready,
            "warnings": list(self.warnings),
        }


def probe_trace_host() -> TraceHostProbe:
    """Inspect the current interpreter, torch runtime, and NVIDIA driver."""

    warnings: list[str] = []
    torch_version = "not-installed"
    cuda_version = _UNAVAILABLE
    accelerator_names: tuple[str, ...] = ()
    bf16_supported = False

    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as error:
        warnings.append(f"torch is unavailable in this interpreter: {error}")
    else:
        torch_version = str(torch.__version__)
        reported_cuda = getattr(getattr(torch, "version", None), "cuda", None)
        if reported_cuda is not None:
            cuda_version = str(reported_cuda)
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            count = int(cuda.device_count())
            accelerator_names = tuple(
                str(cuda.get_device_name(index)) for index in range(count)
            )
            bf16_supported = bool(cuda.is_bf16_supported())
        else:
            warnings.append("torch cannot access a CUDA accelerator")

    smi_names, driver_version, smi_warning = _probe_nvidia_smi()
    if smi_warning is not None:
        warnings.append(smi_warning)
    if smi_names and accelerator_names and len(smi_names) != len(accelerator_names):
        warnings.append(
            "torch and nvidia-smi report different accelerator counts"
        )
    if not accelerator_names and smi_names:
        warnings.append(
            "nvidia-smi sees accelerators that the current torch runtime cannot use"
        )

    runtime = PolicyRuntimeEnvironment(
        python_version=platform.python_version(),
        platform=platform.platform(),
        accelerator=(
            " | ".join(accelerator_names) if accelerator_names else _UNAVAILABLE
        ),
        accelerator_count=len(accelerator_names),
        driver_version=driver_version,
        cuda_version=cuda_version,
        torch_version=torch_version,
    )
    collection_ready = (
        runtime.accelerator_count > 0
        and runtime.torch_version != "not-installed"
        and runtime.cuda_version != _UNAVAILABLE
        and runtime.driver_version != _UNAVAILABLE
    )
    return TraceHostProbe(
        probe_schema_version=TRACE_HOST_PROBE_SCHEMA_VERSION,
        runtime=runtime,
        python_executable=sys.executable,
        driver_accelerators=smi_names,
        bf16_supported=bf16_supported,
        collection_ready=collection_ready,
        warnings=tuple(warnings),
    )


def _probe_nvidia_smi() -> tuple[tuple[str, ...], str, str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return (), _UNAVAILABLE, "nvidia-smi is not available"
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return (), _UNAVAILABLE, f"nvidia-smi probe failed: {error}"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit {completed.returncode}"
        return (), _UNAVAILABLE, f"nvidia-smi probe failed: {detail}"
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    parsed: list[tuple[str, str]] = []
    for row in rows:
        name, separator, driver = row.rpartition(",")
        if not separator or not name.strip() or not driver.strip():
            return (), _UNAVAILABLE, "nvidia-smi returned an unrecognized row"
        parsed.append((name.strip(), driver.strip()))
    if not parsed:
        return (), _UNAVAILABLE, "nvidia-smi reported no accelerators"
    drivers = {driver for _, driver in parsed}
    if len(drivers) != 1:
        return (), _UNAVAILABLE, "nvidia-smi reported mixed driver versions"
    return tuple(name for name, _ in parsed), parsed[0][1], None


def write_trace_host_probe(path: str | Path, probe: TraceHostProbe) -> None:
    """Atomically write one canonical trace-host probe."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(probe.to_dict(), indent=2, sort_keys=True, ensure_ascii=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except OSError as error:
        raise WorkloadTraceError(
            f"cannot write trace host probe {destination}: {error}"
        ) from error


def read_trace_host_probe(path: str | Path) -> TraceHostProbe:
    """Read one strict trace-host probe."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise WorkloadTraceError(
            f"cannot read trace host probe {source}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise WorkloadTraceError(
            f"invalid trace host probe JSON {source}: {error.msg}"
        ) from error
    return parse_trace_host_probe(raw)


def parse_trace_host_probe(raw: Any) -> TraceHostProbe:
    """Parse a strict host probe without allowing undeclared runtime fields."""

    if not isinstance(raw, Mapping):
        raise WorkloadTraceError("trace host probe must be an object")
    required = {
        "probe_schema_version",
        "runtime",
        "python_executable",
        "driver_accelerators",
        "bf16_supported",
        "collection_ready",
        "warnings",
    }
    _require_keys(raw, required, "trace host probe")
    runtime_raw = raw["runtime"]
    if not isinstance(runtime_raw, Mapping):
        raise WorkloadTraceError("trace host probe runtime must be an object")
    _require_keys(
        runtime_raw,
        {
            "python_version",
            "platform",
            "accelerator",
            "accelerator_count",
            "driver_version",
            "cuda_version",
            "torch_version",
        },
        "trace host probe runtime",
    )
    warnings = raw["warnings"]
    if not isinstance(warnings, list) or any(
        not isinstance(warning, str) for warning in warnings
    ):
        raise WorkloadTraceError("trace host probe warnings must be a string array")
    driver_accelerators = raw["driver_accelerators"]
    if not isinstance(driver_accelerators, list) or any(
        not isinstance(name, str) for name in driver_accelerators
    ):
        raise WorkloadTraceError(
            "trace host probe driver_accelerators must be a string array"
        )
    for key in (
        "python_version",
        "platform",
        "accelerator",
        "driver_version",
        "cuda_version",
        "torch_version",
    ):
        if not isinstance(runtime_raw[key], str):
            raise WorkloadTraceError(
                f"trace host probe runtime {key} must be a string"
            )
    count = runtime_raw["accelerator_count"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise WorkloadTraceError(
            "trace host probe runtime accelerator_count must be an integer"
        )
    for key in ("bf16_supported", "collection_ready"):
        if not isinstance(raw[key], bool):
            raise WorkloadTraceError(f"trace host probe {key} must be a boolean")
    version = raw["probe_schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise WorkloadTraceError(
            "trace host probe probe_schema_version must be an integer"
        )
    executable = raw["python_executable"]
    if not isinstance(executable, str):
        raise WorkloadTraceError(
            "trace host probe python_executable must be a string"
        )
    return TraceHostProbe(
        probe_schema_version=version,
        runtime=PolicyRuntimeEnvironment(
            python_version=runtime_raw["python_version"],
            platform=runtime_raw["platform"],
            accelerator=runtime_raw["accelerator"],
            accelerator_count=count,
            driver_version=runtime_raw["driver_version"],
            cuda_version=runtime_raw["cuda_version"],
            torch_version=runtime_raw["torch_version"],
        ),
        python_executable=executable,
        driver_accelerators=tuple(driver_accelerators),
        bf16_supported=raw["bf16_supported"],
        collection_ready=raw["collection_ready"],
        warnings=tuple(warnings),
    )


def _require_keys(raw: Mapping[str, Any], required: set[str], path: str) -> None:
    actual = set(raw)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise WorkloadTraceError(f"{path} fields " + "; ".join(details))
