# ORCHES Reproduction

This repository is an independent, auditable reproduction of the ORCHES
GPU-PIM architecture. It uses AttAcc's modified Ramulator2 as the PIM
foundation and Duplex as a comparison baseline, while keeping ORCHES-specific
configuration, modeling, scheduling, and evaluation code in this repository.
The paper input is identified by DOI and SHA-256 in `third_party.lock`.

## Environment

The Python environment is managed by UV and locked in `uv.lock`:

```bash
uv sync --frozen
```

Third-party source revisions are frozen in `third_party.lock`. Generated
checkouts, traces, and experimental outputs are intentionally excluded from
Git.

Reconstruct missing third-party checkouts and build AttAcc Ramulator2:

```bash
bash scripts/bootstrap.sh
```

Audit an existing environment without modifying it:

```bash
bash scripts/bootstrap.sh --verify-only
```

## M1 Commands

Validate the paper-mapped hardware configurations:

```bash
uv run orches validate-config configs/hardware/agx_orin_32gb.yaml
uv run orches validate-config configs/hardware/orches_pim_32gb.yaml
```

Print all values and provenance as JSON:

```bash
uv run orches validate-config configs/hardware/orches_pim_32gb.yaml --json
```

Run the tests:

```bash
uv run pytest
```

Generate and validate a deterministic development-only TTC trace:

```bash
uv run orches generate-synthetic-trace artifacts/traces/synthetic.jsonl \
  --requests 2 --steps 3 --width 4 --seed 0
uv run orches validate-trace artifacts/traces/synthetic.jsonl --json
```

Synthetic traces exercise replay and simulator code but are not valid paper
evaluation inputs.

Validate a frozen model architecture and run the native PIM smoke benchmark:

```bash
uv run orches validate-model-config configs/models/policy/qwen2.5-1.5b.yaml
uv run orches pim-microbench artifacts/raw/pim-smoke --json
```

The M3 ORCHES implementation is organized by paper technique:

```text
src/orches/scheduler/   Technique 1A/1B assignment and balancing
src/orches/predictor/   Technique 2A/2B prediction and verification
src/orches/memory/      Technique 3 allocation, cache, buffer, and compaction
src/orches/replay.py    schema-v1 synthetic request composition
src/orches/replay_v2.py exact-token, multi-parent request composition
```

`docs/orches-techniques.md` explains the execution sequence and exact paper
correspondence. Schema-v2 replay preserves actual token rounds, materialized KV,
multiple retained beams, and ordered scalar/pairwise selection. These tests
validate control flow but do not replace real workload collection or hardware
calibration.

M4 comparison code uses one common result boundary:

```text
src/orches/baselines/  paper configurations, fairness checks, native adapters
src/orches/metrics/    activity energy, area denominator, and utilization
```

`docs/baseline-contract.md` documents the required fairness fingerprint and
the exact AttAcc/Duplex unit conversions. Native output is never normalized
until its trace, model, precision, capacity, and bandwidth contract matches the
GPU reference.

`DEVELOPMENT.md` is the milestone ledger. `REPRODUCTION_PLAN.md` describes the
full implementation and evaluation plan, and `docs/paper-to-code.md` maps paper
claims to implementation and tests.
