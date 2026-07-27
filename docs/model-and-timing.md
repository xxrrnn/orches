# Model and Timing Contract

## Architecture Profiles

Transformer YAML profiles freeze the Llama/Qwen dimensions needed to expand
operators: hidden and intermediate sizes, decoder layers, query and KV heads,
head dimension, vocabulary, context limit, precision, and embedding tying.
Validation enforces:

```text
hidden_size = num_attention_heads * head_dim
num_attention_heads % num_key_value_heads = 0
```

Policy profiles correspond to ORCHES Sec. 5.1's Llama3.2-1B,
Qwen2.5-1.5B, and Qwen2.5-3B models. PRM names in the paper do not identify
public checkpoint revisions. Their architecture proxies are therefore marked
`ASSUMED`, even where the shape is inherited from a public Qwen/RLHFlow config.
Collected traces must replace proxy identities with the exact checkpoint and
tokenizer revisions actually used.

## Operator DAG

One decode-layer expansion contains input normalization, Q/K/V projections,
RoPE, shared and unique attention-score fragments, one combined softmax,
shared and unique context fragments, output projection, post-attention
normalization, SwiGLU gate/up activation, and down projection.

Each node reports MACs, conventional FLOPs, read bytes, write bytes, and
topological dependencies. One MAC is two conventional FLOPs. Shared KV is read
once for batched branch queries; unique KV storage is branch-private and scales
with branch width. This makes shared attention arithmetic intensity improve
with width while unique attention remains approximately constant, matching
Sec. 3.1's motivation.

## Two Timing Conventions

The assignment equations and the device roofline intentionally use different
types and function names.

`paper_gpu_linear_time`, `paper_pim_linear_time`, and
`paper_coprocessed_linear_time` implement ORCHES Eq. (1)-(4) using MAC/s and
tensor-elements/s. They follow the paper's operation counts exactly and do not
insert a factor of two. Their terms are additive as printed in the equations.

`gpu_roofline_time` uses conventional FLOP/s and byte/s. It follows the
Duplex-style analytical execution rule:

```text
T = max(FLOPs / effective_FLOP_per_s, bytes / effective_byte_per_s)
  + launch_overhead
  + synchronization_overhead
```

The SoC sensitivity profile scales only memory byte/s. It does not change
compute throughput or fixed overhead.

No `orin_calibrated` AGX Orin rate profile is committed yet. The main
paper-method reproduction may use an AttAcc-style analytical profile whose peak
and utilization provenance remains explicit. Official peak specifications are
bounds, not achieved measurements. The two evidence profiles and the reason a
5070 Ti workload trace does not supply Orin timing are defined in
`calibration.md` and `trace-collection-plan.md`.
