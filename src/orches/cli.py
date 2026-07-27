"""Command-line interface for ORCHES reproducibility utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from .config import load_hardware_config
from .errors import ConfigurationError, OrchesInputError, SimulationError
from .hardware import PimHardwareConfig
from .models import load_transformer_config
from .sim import (
    generate_mac_microbenchmark,
    render_attacc_config,
    run_attacc_ramulator,
    write_attacc_config,
    write_attacc_trace,
)
from .workload import (
    Modality,
    QuestionLengthBucket,
    SyntheticTraceConfig,
    generate_synthetic_traces,
    read_jsonl,
    trace_sha256,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIM_CONFIG = PROJECT_ROOT / "configs/hardware/orches_pim_32gb.yaml"
DEFAULT_RAMULATOR = (
    PROJECT_ROOT / "third_party/attacc_simulator/ramulator2/ramulator2"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orches",
        description="Validate and inspect ORCHES reproduction inputs.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate-config",
        help="Validate one provenance-aware hardware YAML file.",
    )
    validate.add_argument("path", type=Path, help="Path to the hardware YAML file")
    validate.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable validation summary.",
    )

    validate_trace = commands.add_parser(
        "validate-trace",
        help="Validate and summarize one versioned TTC JSONL trace.",
    )
    validate_trace.add_argument("path", type=Path, help="Path to the trace JSONL file")
    validate_trace.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable trace summary.",
    )

    validate_model = commands.add_parser(
        "validate-model-config",
        help="Validate one Transformer architecture YAML file.",
    )
    validate_model.add_argument("path", type=Path, help="Path to the model YAML file")
    validate_model.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable model summary.",
    )

    synthetic = commands.add_parser(
        "generate-synthetic-trace",
        help="Generate a deterministic development trace (not evaluation data).",
    )
    synthetic.add_argument("output", type=Path, help="Destination JSONL path")
    synthetic.add_argument("--requests", type=int, default=1)
    synthetic.add_argument("--steps", type=int, default=3)
    synthetic.add_argument("--width", type=int, default=4)
    synthetic.add_argument("--prompt-tokens", type=int, default=128)
    synthetic.add_argument("--min-generated-tokens", type=int, default=4)
    synthetic.add_argument("--max-generated-tokens", type=int, default=12)
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.add_argument(
        "--modality",
        choices=[modality.value for modality in Modality],
        default=Modality.TEXT.value,
    )
    synthetic.add_argument("--image-tokens", type=int, default=0)
    synthetic.add_argument(
        "--question-length-bucket",
        choices=[bucket.value for bucket in QuestionLengthBucket],
    )

    pim_microbench = commands.add_parser(
        "pim-microbench",
        help="Generate and run a small AttAcc-compatible ORCHES PIM trace.",
    )
    pim_microbench.add_argument("output_dir", type=Path)
    pim_microbench.add_argument(
        "--hardware",
        type=Path,
        default=DEFAULT_PIM_CONFIG,
    )
    pim_microbench.add_argument(
        "--ramulator",
        type=Path,
        default=DEFAULT_RAMULATOR,
    )
    pim_microbench.add_argument("--mac-rounds", type=int, default=1)
    pim_microbench.add_argument("--json", action="store_true")
    return parser


def _print_human_summary(summary: dict[str, object]) -> None:
    print(f"valid: {summary['name']} ({summary['kind']})")
    print("derived:")
    derived = summary["derived"]
    if not isinstance(derived, Mapping):
        raise RuntimeError("hardware summary has an invalid derived section")
    for key, value in derived.items():
        print(f"  {key}: {value}")
    print("evidence:")
    evidence = summary["evidence"]
    if not isinstance(evidence, Mapping):
        raise RuntimeError("hardware summary has an invalid evidence section")
    for key, value in evidence.items():
        print(f"  {key}: {value}")


def _trace_summary(path: Path) -> dict[str, object]:
    traces = read_jsonl(path)
    return {
        "path": str(path),
        "sha256": trace_sha256(path),
        "schema_versions": sorted({trace.trace_schema_version for trace in traces}),
        "requests": len(traces),
        "steps": sum(len(trace.steps) for trace in traces),
        "candidates": sum(trace.candidate_count for trace in traces),
        "generated_tokens": sum(trace.generated_tokens for trace in traces),
        "selected_path_tokens": sum(trace.selected_path_tokens for trace in traces),
        "modalities": sorted({trace.modality.value for trace in traces}),
        "source_kinds": sorted(
            {trace.provenance.source_kind.value for trace in traces}
        ),
        "beam_sizes": sorted({getattr(trace, "beam_size", 1) for trace in traces}),
    }


def _print_trace_summary(summary: dict[str, object]) -> None:
    print(f"valid trace: {summary['path']}")
    for key in (
        "sha256",
        "schema_versions",
        "requests",
        "steps",
        "candidates",
        "generated_tokens",
        "selected_path_tokens",
        "modalities",
        "source_kinds",
        "beam_sizes",
    ):
        print(f"  {key}: {summary[key]}")


def _print_mapping_summary(title: str, summary: dict[str, object]) -> None:
    print(title)
    for key, value in summary.items():
        if isinstance(value, Mapping):
            print(f"{key}:")
            for nested_key, nested_value in value.items():
                print(f"  {nested_key}: {nested_value}")
        else:
            print(f"{key}: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ORCHES CLI and return a process exit status."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_hardware_config(args.path)
            summary = config.summary()
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                _print_human_summary(summary)
            return 0
        if args.command == "validate-trace":
            summary = _trace_summary(args.path)
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                _print_trace_summary(summary)
            return 0
        if args.command == "validate-model-config":
            summary = load_transformer_config(args.path).summary()
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                _print_mapping_summary(
                    f"valid model: {summary['name']} ({summary['role']})",
                    summary["derived"],
                )
            return 0
        if args.command == "generate-synthetic-trace":
            bucket = (
                QuestionLengthBucket(args.question_length_bucket)
                if args.question_length_bucket is not None
                else None
            )
            config = SyntheticTraceConfig(
                request_count=args.requests,
                step_count=args.steps,
                search_width=args.width,
                prompt_tokens=args.prompt_tokens,
                min_generated_tokens=args.min_generated_tokens,
                max_generated_tokens=args.max_generated_tokens,
                seed=args.seed,
                modality=Modality(args.modality),
                image_tokens=args.image_tokens,
                question_length_bucket=bucket,
            )
            write_jsonl(args.output, generate_synthetic_traces(config))
            summary = _trace_summary(args.output)
            _print_trace_summary(summary)
            print("  evaluation_eligible: False")
            return 0
        if args.command == "pim-microbench":
            hardware = load_hardware_config(args.hardware)
            if not isinstance(hardware, PimHardwareConfig):
                raise ConfigurationError("pim-microbench requires a PIM hardware config")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            trace_path = args.output_dir / "pim_mac.trace"
            config_path = args.output_dir / "ramulator.yaml"
            log_path = args.output_dir / "log/cmd.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            commands = generate_mac_microbenchmark(
                hardware,
                mac_rounds=args.mac_rounds,
            )
            write_attacc_trace(trace_path, commands)
            write_attacc_config(
                config_path,
                render_attacc_config(
                    hardware,
                    trace_path=trace_path,
                    log_path=log_path,
                ),
            )
            result = run_attacc_ramulator(args.ramulator, config_path)
            summary = {
                "hardware": hardware.name,
                "trace_path": str(trace_path),
                "config_path": str(config_path),
                "command_count": len(commands),
                "memory_system_cycles": result.memory_system_cycles,
                "counters": result.counters,
            }
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                _print_mapping_summary("PIM microbenchmark complete", summary)
            return 0
    except OrchesInputError as error:
        print(f"input error: {error}", file=sys.stderr)
        return 2
    except SimulationError as error:
        print(f"simulation error: {error}", file=sys.stderr)
        return 3

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
