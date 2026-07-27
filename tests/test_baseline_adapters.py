from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from orches.baselines import (
    BASELINES,
    FIGURE_12_BASELINES,
    BaselineKind,
    BaselineMetrics,
    BaselineRunResult,
    FairnessContract,
    PlacementStrategy,
    RunStatus,
    compare_baselines,
    parse_attacc_csv,
    parse_duplex_csv,
)
from orches.baselines.adapters import (
    ATTACC_ENERGY_COLUMNS,
    DUPLEX_ENERGY_COLUMNS,
    DUPLEX_TIME_COLUMNS,
)
from orches.errors import ConfigurationError


def contract() -> FairnessContract:
    return FairnessContract(
        trace_sha256="a" * 64,
        request_ids=("request-0",),
        policy_model="Qwen2.5-3B",
        policy_revision="policy-rev",
        small_prm_model="Qwen2.5-1.5B-PRM",
        small_prm_revision="small-rev",
        large_prm_model="Qwen2.5-7B-PRM",
        large_prm_revision="large-rev",
        tokenizer="Qwen-tokenizer",
        tokenizer_revision="tokenizer-rev",
        weight_bytes_per_element=2,
        activation_bytes_per_element=2,
        gpu_count=1,
        soc_bandwidth_bytes_per_s=204.8e9,
        pim_capacity_bytes=32 * 1024**3,
        hardware_contract_sha256="b" * 64,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_paper_ablation_definitions_have_explicit_switches() -> None:
    assert FIGURE_12_BASELINES == (
        BaselineKind.GPU,
        BaselineKind.ATTACC,
        BaselineKind.ORCHES_A,
        BaselineKind.ORCHES_B,
        BaselineKind.ORCHES_C,
    )
    assert BASELINES[BaselineKind.ORCHES_A].placement is PlacementStrategy.ALL_ON_PIM
    assert BASELINES[BaselineKind.ORCHES_B].techniques.t1a
    assert not BASELINES[BaselineKind.ORCHES_B].techniques.t1b
    assert BASELINES[BaselineKind.ORCHES_C].techniques.t1b
    assert not BASELINES[BaselineKind.ORCHES_C].techniques.t2a


def test_fairness_fingerprint_is_stable_and_sensitive() -> None:
    original = contract()
    same = contract()
    lower_bandwidth = replace(original, soc_bandwidth_bytes_per_s=153.6e9)

    assert original.fingerprint == same.fingerprint
    assert original.fingerprint != lower_bandwidth.fingerprint


def test_attacc_adapter_converts_ms_nj_and_capacity(tmp_path: Path) -> None:
    row: dict[str, object] = {
        "cap": 32,
        "required_cap": 1024,
        "g_time (ms)": 12.5,
        "g_energy (nJ)": 2500,
    }
    row.update(
        {
            column: index + 1
            for index, column in enumerate(ATTACC_ENERGY_COLUMNS)
        }
    )
    path = tmp_path / "attacc.csv"
    write_csv(path, [row])

    result = parse_attacc_csv(
        path,
        contract=contract(),
        source_revision="c600051",
    )

    assert result.status is RunStatus.SUCCESS
    assert result.metrics is not None
    assert result.metrics.latency_s == pytest.approx(0.0125)
    assert result.metrics.energy_j == pytest.approx(2.5e-6)
    assert result.metrics.peak_memory_bytes == 1024
    assert result.components["g_dram_energy"] == pytest.approx(1e-9)


def test_attacc_adapter_preserves_oom_result(tmp_path: Path) -> None:
    row: dict[str, object] = {
        "cap": 1,
        "required_cap": 2 * 1024**3,
        "g_time (ms)": 1,
        "g_energy (nJ)": 1,
    }
    row.update({column: 0 for column in ATTACC_ENERGY_COLUMNS})
    path = tmp_path / "attacc-oom.csv"
    write_csv(path, [row])

    result = parse_attacc_csv(
        path,
        contract=contract(),
        source_revision="c600051",
    )

    assert result.status is RunStatus.OOM
    assert result.metrics is None
    assert result.error is not None


def duplex_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "iter_info": 1,
        "type": "t2t",
        "time": 2_000_000,
        "latency": 2_000_000,
        "OOM": 0,
        "total_energy": 1000,
    }
    row.update({column: 1 for column in DUPLEX_ENERGY_COLUMNS})
    row.update({column: 10 for column in DUPLEX_TIME_COLUMNS})
    row.update(overrides)
    return row


def test_duplex_adapter_uses_e2e_latency_and_sums_activity_energy(
    tmp_path: Path,
) -> None:
    activity = duplex_row()
    e2e = duplex_row(
        iter_info=0,
        type="e2e",
        latency=3_000_000,
        total_energy=0,
    )
    path = tmp_path / "duplex.csv"
    write_csv(path, [activity, e2e])

    result = parse_duplex_csv(
        path,
        contract=contract(),
        source_revision="4192527",
    )

    assert result.status is RunStatus.SUCCESS
    assert result.metrics is not None
    assert result.metrics.latency_s == pytest.approx(0.003)
    assert result.metrics.energy_j == pytest.approx(1e-6)
    assert result.components["energy.act_energy"] == pytest.approx(1e-9)
    assert result.components["time.atten_gen"] == pytest.approx(10e-9)


def test_comparison_rejects_mixed_contracts() -> None:
    gpu = BaselineRunResult(
        BaselineKind.GPU,
        contract(),
        RunStatus.SUCCESS,
        "gpu-calibration",
        BaselineMetrics(latency_s=2.0, energy_j=4.0),
    )
    attacc = BaselineRunResult(
        BaselineKind.ATTACC,
        replace(contract(), soc_bandwidth_bytes_per_s=153.6e9),
        RunStatus.SUCCESS,
        "c600051",
        BaselineMetrics(latency_s=1.0, energy_j=2.0),
    )

    with pytest.raises(ConfigurationError, match="fairness-contract"):
        compare_baselines(
            (gpu, attacc),
            required=(BaselineKind.GPU, BaselineKind.ATTACC),
        )


def test_comparison_normalizes_success_and_retains_failure_row() -> None:
    gpu = BaselineRunResult(
        BaselineKind.GPU,
        contract(),
        RunStatus.SUCCESS,
        "gpu-calibration",
        BaselineMetrics(latency_s=2.0, energy_j=4.0),
    )
    attacc = BaselineRunResult(
        BaselineKind.ATTACC,
        contract(),
        RunStatus.SUCCESS,
        "c600051",
        BaselineMetrics(latency_s=1.0, energy_j=2.0),
    )
    duplex = BaselineRunResult(
        BaselineKind.DUPLEX,
        contract(),
        RunStatus.OOM,
        "4192527",
        error="native OOM",
    )

    rows = compare_baselines(
        (gpu, attacc, duplex),
        required=(BaselineKind.GPU, BaselineKind.ATTACC, BaselineKind.DUPLEX),
    )

    assert rows[0].speedup_vs_gpu == 1.0
    assert rows[1].speedup_vs_gpu == 2.0
    assert rows[1].energy_efficiency_vs_gpu == 2.0
    assert rows[2].status is RunStatus.OOM
    assert rows[2].speedup_vs_gpu is None
    assert rows[2].error == "native OOM"


def test_native_adapter_rejects_nonfinite_value(tmp_path: Path) -> None:
    row: dict[str, object] = {
        "cap": 32,
        "required_cap": 1024,
        "g_time (ms)": "nan",
        "g_energy (nJ)": 1,
    }
    row.update({column: 0 for column in ATTACC_ENERGY_COLUMNS})
    path = tmp_path / "attacc-nan.csv"
    write_csv(path, [row])

    with pytest.raises(ConfigurationError, match="finite"):
        parse_attacc_csv(
            path,
            contract=contract(),
            source_revision="c600051",
        )
