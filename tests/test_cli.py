from __future__ import annotations

import json
from pathlib import Path

import pytest

from orches.cli import main


PROJECT_ROOT = Path(__file__).parents[1]
GPU_CONFIG = PROJECT_ROOT / "configs/hardware/agx_orin_32gb.yaml"
PIM_CONFIG = PROJECT_ROOT / "configs/hardware/orches_pim_32gb.yaml"
MODEL_CONFIG = PROJECT_ROOT / "configs/models/policy/qwen2.5-1.5b.yaml"
RAMULATOR = PROJECT_ROOT / "third_party/attacc_simulator/ramulator2/ramulator2"


def test_validate_config_human_output(capsys) -> None:
    status = main(["validate-config", str(PIM_CONFIG)])

    captured = capsys.readouterr()
    assert status == 0
    assert "valid: orches_pim_32gb (pim)" in captured.out
    assert "bank_count: 2048" in captured.out
    assert captured.err == ""


def test_validate_config_json_contains_provenance(capsys) -> None:
    status = main(["validate-config", str(GPU_CONFIG), "--json"])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert status == 0
    assert summary["derived"]["bandwidth_points_gb_per_s"] == [
        204.8,
        153.6,
        102.4,
    ]
    assert summary["parameters"]["gpu.memory_bandwidth_gb_per_s"]["status"] == "PAPER"
    assert "ORCHES Sec. 5.1" in summary["parameters"][
        "gpu.memory_bandwidth_gb_per_s"
    ]["source"]


def test_validate_config_reports_user_error(tmp_path, capsys) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("kind: pim\n", encoding="utf-8")

    status = main(["validate-config", str(invalid)])

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert "input error:" in captured.err


def test_generate_and_validate_synthetic_trace(tmp_path, capsys) -> None:
    trace_path = tmp_path / "synthetic.jsonl"

    generate_status = main(
        [
            "generate-synthetic-trace",
            str(trace_path),
            "--requests",
            "2",
            "--steps",
            "2",
            "--width",
            "3",
            "--seed",
            "17",
        ]
    )
    generated = capsys.readouterr()
    validate_status = main(["validate-trace", str(trace_path), "--json"])
    validated = capsys.readouterr()
    summary = json.loads(validated.out)

    assert generate_status == 0
    assert validate_status == 0
    assert "evaluation_eligible: False" in generated.out
    assert summary["requests"] == 2
    assert summary["steps"] == 4
    assert summary["candidates"] == 12
    assert summary["source_kinds"] == ["synthetic"]


def test_synthetic_vision_cli_requires_metadata(tmp_path, capsys) -> None:
    status = main(
        [
            "generate-synthetic-trace",
            str(tmp_path / "vision.jsonl"),
            "--modality",
            "vision",
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert "require image tokens and a length bucket" in captured.err


def test_validate_model_config_json(capsys) -> None:
    status = main(["validate-model-config", str(MODEL_CONFIG), "--json"])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert status == 0
    assert summary["name"] == "Qwen2.5-1.5B-Instruct"
    assert summary["derived"]["kv_hidden_size"] == 256


@pytest.mark.integration
def test_pim_microbench_cli(tmp_path, capsys) -> None:
    if not RAMULATOR.is_file():
        pytest.skip("run scripts/bootstrap.sh before the native integration test")

    status = main(
        [
            "pim-microbench",
            str(tmp_path),
            "--hardware",
            str(PIM_CONFIG),
            "--ramulator",
            str(RAMULATOR),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert status == 0
    assert summary["memory_system_cycles"] > 0
    assert summary["counters"]["total_num_pim_mac_all_bank_requests"] == 32
