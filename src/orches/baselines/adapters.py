"""Strict adapters for frozen AttAcc and Duplex native result schemas."""

from __future__ import annotations

import csv
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Mapping

from ..errors import ConfigurationError
from .definitions import BaselineKind
from .results import (
    BaselineMetrics,
    BaselineRunResult,
    FairnessContract,
    RunStatus,
)


ATTACC_ENERGY_COLUMNS = (
    "g_dram_energy",
    "g_l2_energy",
    "g_l1_energy",
    "g_reg_energy",
    "g_alu_energy",
    "g_fc_mem_energy",
    "g_fc_comp_energy",
    "g_attn_mem_energy",
    "g_attn_comp_energy",
    "g_etc_mem_energy",
    "g_etc_comp_energy",
    "g_comm_energy",
)

DUPLEX_ENERGY_COLUMNS = (
    "act_energy",
    "read_energy",
    "write_energy",
    "all_act_energy",
    "all_read_energy",
    "all_write_energy",
    "mac_energy",
)

DUPLEX_TIME_COLUMNS = (
    "qkvgen",
    "q_down_proj",
    "kv_down_proj",
    "kr_proj",
    "q_up_proj",
    "qr_proj",
    "kv_up_proj",
    "tr_k_up_proj",
    "v_up_proj",
    "atten_sum",
    "atten_gen",
    "o_proj",
    "ffn",
    "expert_ffn",
    "communication",
    "rope",
    "layernorm",
    "residual",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise ConfigurationError(
            f"cannot read native result {path}: {error}"
        ) from error
    if not rows:
        raise ConfigurationError(f"native result {path} contains no data rows")
    return rows


def _number(row: Mapping[str, str], column: str) -> float:
    if column not in row or row[column] is None or not str(row[column]).strip():
        raise ConfigurationError(f"native result is missing column {column!r}")
    try:
        value = float(row[column])
    except ValueError as error:
        raise ConfigurationError(
            f"native result column {column!r} is not numeric"
        ) from error
    if not isfinite(value) or value < 0:
        raise ConfigurationError(
            f"native result column {column!r} must be finite and non-negative"
        )
    return value


def parse_attacc_csv(
    path: Path,
    *,
    contract: FairnessContract,
    source_revision: str,
    row_index: int = -1,
) -> BaselineRunResult:
    """Convert one AttAcc CSV row from ms/nJ to seconds/joules."""

    rows = _read_rows(path)
    try:
        row = rows[row_index]
    except IndexError as error:
        raise ConfigurationError(
            f"AttAcc row index {row_index} is out of range"
        ) from error
    capacity_bytes = int(_number(row, "cap") * 1024**3)
    required_bytes = int(_number(row, "required_cap"))
    if required_bytes > capacity_bytes:
        return BaselineRunResult(
            baseline=BaselineKind.ATTACC,
            contract=contract,
            status=RunStatus.OOM,
            source_revision=source_revision,
            error=(
                f"required {required_bytes} bytes exceeds native capacity "
                f"{capacity_bytes} bytes"
            ),
        )
    components = {
        column: _number(row, column) * 1e-9
        for column in ATTACC_ENERGY_COLUMNS
    }
    return BaselineRunResult(
        baseline=BaselineKind.ATTACC,
        contract=contract,
        status=RunStatus.SUCCESS,
        source_revision=source_revision,
        metrics=BaselineMetrics(
            latency_s=_number(row, "g_time (ms)") * 1e-3,
            energy_j=_number(row, "g_energy (nJ)") * 1e-9,
            peak_memory_bytes=required_bytes,
        ),
        components=components,
    )


def parse_duplex_csv(
    path: Path,
    *,
    contract: FairnessContract,
    source_revision: str,
) -> BaselineRunResult:
    """Summarize Duplex iteration/e2e rows from ns/nJ into SI units."""

    rows = _read_rows(path)
    if any(_number(row, "OOM") != 0 for row in rows):
        return BaselineRunResult(
            baseline=BaselineKind.DUPLEX,
            contract=contract,
            status=RunStatus.OOM,
            source_revision=source_revision,
            error="Duplex native result reports OOM",
        )
    activity_rows = [row for row in rows if row.get("iter_info") == "1"]
    if not activity_rows:
        raise ConfigurationError("Duplex result has no iter_info=1 activity rows")
    request_rows = [row for row in rows if row.get("type") == "e2e"]
    if request_rows:
        latency_ns = fmean(_number(row, "latency") for row in request_rows)
    else:
        latency_ns = max(_number(row, "time") for row in activity_rows)
    components = {
        f"energy.{column}": sum(_number(row, column) for row in activity_rows)
        * 1e-9
        for column in DUPLEX_ENERGY_COLUMNS
    }
    components.update(
        {
            f"time.{column}": sum(_number(row, column) for row in activity_rows)
            * 1e-9
            for column in DUPLEX_TIME_COLUMNS
        }
    )
    return BaselineRunResult(
        baseline=BaselineKind.DUPLEX,
        contract=contract,
        status=RunStatus.SUCCESS,
        source_revision=source_revision,
        metrics=BaselineMetrics(
            latency_s=latency_ns * 1e-9,
            energy_j=sum(
                _number(row, "total_energy") for row in activity_rows
            )
            * 1e-9,
        ),
        components=components,
    )
