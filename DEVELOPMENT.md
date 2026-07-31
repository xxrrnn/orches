# ORCHES Development Log

## M0: Compute-Optimal-TTS local environment and PRM startup

### Goal

Run the unmodified Compute-Optimal-TTS search algorithm locally so that it can
later generate real token-level traces for AttAcc, Duplex, and ORCHES. This is
only a local smoke configuration for an RTX 5070 Ti (16 GiB); it is not a
paper-scale evaluation configuration.

### Reproducible environment

- Upstream repository: `third_party/compute-optimal-tts`
- Upstream revision: `0ee2578af1f8d6cac445c9c4c72780528bb94556`
- Environment definition: `environments/compute-optimal-tts/rtx5070ti`
- Python: 3.10
- GPU runtime: PyTorch 2.7.0 with CUDA 12.8 wheels
- Inference server: vLLM 0.9.1
- Model cache: `/home/xrn/.cache/huggingface/orches-tts`

The upstream `src/requirements.txt` pins PyTorch 2.5.1, Transformers 4.47.0,
and vLLM 0.6.4.post1. Those versions predate Blackwell support required by the
RTX 5070 Ti. The local UV environment therefore pins a newer compatible stack.
The search algorithm and its evaluation parameters remain unchanged. Small
compatibility fixes are documented below where the upstream worker API assumes
older runtime libraries.

### Model download result

The PRM `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` was downloaded completely
(`pytorch_model.bin`, 3.09 GB). `HF_ENDPOINT=https://hf-mirror.com` could not
be used: the mirror returns HTTP 308 to `huggingface.co`, while
`huggingface_hub` requires file metadata on the initial response. The working
endpoint is `HF_ENDPOINT=https://huggingface.co`.

### PRM runtime compatibility fix

The downloaded legacy checkpoint contains 341 tensors in two PyTorch zip
storage records; the largest record is 3.087 GB. Sequentially reading the file
took 0.30 seconds, while both ordinary `torch.load` and the PyTorch 2.7
weights-only loader remained in `get_storage_from_record` for more than two
minutes. This rules out network, disk throughput, GPU detection, and host RAM
as the source of the startup delay.

`torch.load(..., mmap=True)` loaded the same checkpoint metadata in 1.32
seconds. The patch is deliberately limited to the Skywork PRM path:

- `reward_model_worker.py` replaces Transformers' legacy `.bin` state-dict
  reader with an mmap reader. Safetensors behavior is unchanged.
- `modeling_base.py` uses mmap for the wrapper's second legacy read. The
  upstream wrapper otherwise reads the full checkpoint twice, although the
  second pass only initializes the small value head.
- `modeling_base.py` binds `prepare_inputs_for_generation` only when the
  underlying model supplies it. The current `Qwen2ForRewardModel` is a reward
  model and does not expose a generation method; PRM scoring does not require
  one.

These changes preserve checkpoint weights, reward scoring semantics, search
parameters, token accounting, and all inference trace events. They only make
the upstream 2025 source compatible with the Blackwell-capable 2026 runtime.

### Verification result

The configured venv reports `torch 2.7.0+cu128`, CUDA available, and one GPU.
With a controller on port 10014, the patched PRM registered successfully and
started its FastChat worker at `http://127.0.0.1:10081` in about eight seconds.
No commit has been made.

## M1: Local TTS functional path

### Compatibility fixes

The local runtime uses vLLM 0.9.1, while the upstream policy worker expects
the vLLM 0.6.x `AsyncLLMEngine.engine` wrapper. The policy worker now supports
both layouts:

- `vllm_worker.py` retains the upstream tokenizer and model-config lookup when
  `llm_engine.engine` exists.
- For vLLM 0.9.x, it loads the same Hugging Face tokenizer directly and reads
  `llm_engine.model_config`. vLLM still performs all generation and KV-cache
  management; the wrapper uses this tokenizer only for EOS handling and token
  counts.
- The vLLM 0.9.1 CLI uses Python 3.13-only argparse metadata. The existing
  Python 3.10 compatibility shim drops only that metadata while preserving all
  argument values.

Ray local mode selected a `10.129.*` VPN route in WSL and could not connect to
its own GCS process. The `--local 1` branch now starts Ray with
`_node_ip_address` set from `ORCHES_RAY_LOCAL_IP`, defaulting to `127.0.0.1`.
Distributed runs (`--local 0`) are unchanged.

### vLLM limitation on RTX 5070 Ti

The patched policy worker loads, creates its KV cache, captures CUDA graphs,
and registers with FastChat. Its first generation then fails in the packaged
FlashAttention-2 extension with `CUDA error: no kernel image is available for
execution on the device`. Forcing either `TORCH_SDPA` or `XFORMERS` does not
solve this with vLLM 0.9.1: SDPA is unsupported by its V1 engine and the
packaged XFormers binary also lacks a compatible kernel.

This is a vLLM binary-kernel limitation, not an ORCHES/TTS algorithm error.
The local vLLM path requires either a newer vLLM wheel with RTX 50 support or
a source build of the attention kernels for the local GPU. The 5090 deployment
must independently verify its vLLM wheel and CUDA-kernel support before being
used for performance or power traces.

### Functional fallback and end-to-end result

For local functional validation only, the upstream Transformers/FastChat
`model_worker` runs `Qwen/Qwen2.5-0.5B-Instruct` at the upstream single-GPU
policy port `10082`; the Skywork PRM remains at port `10081`. This preserves
the FastChat request protocol, token outputs, PRM scores, and TTS beam-search
logic, but does not represent vLLM scheduling, KV-cache allocation, latency,
or power behavior.

The evaluator also requires loopback proxy bypass because Python `requests`
does not interpret the host environment's `NO_PROXY=127.*` pattern. The local
command clears proxy variables and sets `NO_PROXY` to `127.0.0.1`, `localhost`,
and `0.0.0.0`.

The following end-to-end configuration exited with status zero:

- Dataset: AIME24
- Method: `beam_search`
- Beam size: `--num_sequence 2`
- Maximum branch width: `--tree_max_width 4`
- Maximum depth: `--tree_max_depth 2`
- Local actor count: one (`--local 1 --num_worker 1`)
- Smoke completion bound: `--max_new_tokens 64`
- Artifacts:
  `/tmp/orches-tts-smoke/AIME24_beam_search/Qwen2.5-0.5B-Instruct/Skywork-o1-Open-PRM-Qwen-2.5-1.5B/2_4_2/`

The successful run loaded AIME24, generated policy actions twice, called the
PRM once, completed search for question 0, and wrote `config.json`,
`question_0/record_0.jsonl`, and `avg_result.json`. Existing interrupted-run
state caused that invocation to resume only question 0; a fresh output
directory is required for the full 30-question set. The upstream
`--question_max_num` argument is parsed but not applied to the dataset.

### Next verification

1. Preserve this functional fallback only for local integration tests.
2. Build or select a vLLM runtime with verified RTX 50 attention kernels on
   the 5090 host, then repeat policy generation before producing performance
   traces.
3. Add token and KV-cache trace collection only after that uninstrumented
   vLLM-backed run succeeds.

No `git add` or `git commit` has been performed.

## M3: Reproducible RTX 5090 vLLM deployment (Path 1)

### Chosen deployment path

Path 1 transfers a vLLM wheel built locally for Blackwell rather than compiling
vLLM on the remote server. This is the appropriate path for the 5090 host:
the host has a sufficiently new NVIDIA driver but no `nvcc`, and the user does
not have sudo access. A CUDA Toolkit installation on that host is therefore
not required to run this wheel.

The wheel was built from vLLM `v0.9.1` source commit
`b6553be1bc75f046b00046a4ad7576364d03c835` with CUDA Toolkit 12.8 and these
architecture settings:

```
VLLM_TARGET_DEVICE=cuda
TORCH_CUDA_ARCH_LIST=12.0
CMAKE_CUDA_ARCHITECTURES=120
MAX_JOBS=4
```

`VLLM_USE_PRECOMPILED` was unset during the build. The resulting file is
`artifacts/vllm-sm120/vllm-0.9.1-cp310-cp310-linux_x86_64.whl`, SHA256
`937bd9dbfaadaf816c2857c7ae0e1b73bfe805a0c43af030ceb034f4132da74a`.
`cuobjdump` confirms its FlashAttention extension contains an `sm_120` image
compiled with CUDA 12.8.

The wheel is an ignored local artifact, not a Git-tracked binary. The planned
deployment script must transfer it and verify this SHA256 before `uv sync`.

### Locked remote environment

A dedicated reproducible environment definition now exists at
`environments/compute-optimal-tts/rtx5090/`. Its `pyproject.toml` pins the
same TTS dependency set as the local environment and resolves `vllm==0.9.1`
from the verified local wheel. Its `uv.lock` pins the exact wheel hash and
the CUDA 12.8 PyTorch family:

- `torch==2.7.0+cu128`
- `torchaudio==2.7.0+cu128`
- `torchvision==0.22.0+cu128`
- `transformers==4.52.4`
- `vllm==0.9.1` from the `sm_120` wheel above

On the remote 5090 host, `uv sync --frozen --python /usr/bin/python3.10`
completed using this lock. Import verification reported PyTorch
`2.7.0+cu128`, torchaudio `2.7.0+cu128`, Transformers `4.52.4`, and vLLM
`0.9.1`.

The remote host cannot reach Hugging Face directly. Its existing loopback
HTTP proxy at `127.0.0.1:7898` works for model downloads, so first-time
downloads require `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` set to that
address. Local vLLM/TTS endpoints must also set `NO_PROXY` and `no_proxy` to
`127.0.0.1,localhost`; otherwise a local worker request can be routed through
the proxy. For vLLM process-group initialization, retain:

```
VLLM_HOST_IP=127.0.0.1
GLOO_SOCKET_IFNAME=lo
NCCL_SOCKET_IFNAME=lo
```

### Actual RTX 5090 inference result

To make the test independent of slow external model download, the verified
local Hugging Face cache for `Qwen/Qwen2.5-0.5B-Instruct` was synchronized to
the remote cache and vLLM was launched in offline mode with
`--max-model-len 512 --enforce-eager`. The remote V1 engine explicitly logged
`Using Flash Attention backend on V1 engine`, loaded the model in 0.39 seconds,
and created 2,253,392 GPU KV-cache token slots.

The server then processed this real loopback request:

```
POST /v1/completions
prompt: "1+1="
max_tokens: 4
temperature: 0
```

It returned HTTP 200 with generated text beginning `2` and usage
`prompt_tokens=4`, `completion_tokens=4`. The test server was stopped and all
remote GPUs were verified free afterward.

This proves that Path 1 can run actual Blackwell vLLM inference on the 5090;
it does not yet prove the full Compute-Optimal-TTS system. The remaining
runtime milestone is to start the FastChat controller, the Skywork PRM worker,
and the Compute-Optimal-TTS policy vLLM worker under this environment, then
run the requested AIME beam-2 / maximum-branch-4 smoke job. Only after that
will trace instrumentation be added.

No `git add` or `git commit` has been performed.

## M2: RTX 50 vLLM attention compatibility

### Local CUDA and FlashAttention result

The WSL host originally exposed only CUDA Toolkit 11.5, which cannot compile
Blackwell `sm_120` kernels. CUDA Toolkit 12.8 is now installed alongside the
old toolkit. The shell configuration selects it explicitly through
`CUDA_HOME=/usr/local/cuda-12.8`, `CUDACXX`, and `PATH`; it does not globally
override `LD_LIBRARY_PATH`, so PyTorch's `cu128` runtime remains authoritative.

The active TTS venv reports `vllm==0.9.1`. Inspection of its installed
`vllm_flash_attn/_vllm_fa2_C.abi3.so` with CUDA 12.8 `cuobjdump` shows an
`sm_120` ELF image compiled with Toolkit 12.8. This replaces the prior
prebuilt extension that failed at first generation with `no kernel image is
available for execution on the device`.

The vLLM 0.9.1 packaging switch is presence-based: setting
`VLLM_USE_PRECOMPILED=0` still enables its precompiled-wheel path because the
nonempty string `"0"` is truthy. Reproducible deployment must therefore
either leave that variable unset for a source build or record the exact wheel
file and SHA256. The current binary is functionally validated, but its wheel
provenance is not yet frozen.

### WSL local-network correction

The first API-server attempt stalled before model load. PyTorch c10d tried to
connect to stale VPN address `10.129.164.7`, because the WSL hostname resolved
only to IPv6 addresses. A server launched with the following process-local
variables initializes correctly without editing `/etc/hosts`:

```
VLLM_HOST_IP=127.0.0.1
GLOO_SOCKET_IFNAME=lo
NCCL_SOCKET_IFNAME=lo
```

`HF_ENDPOINT=https://huggingface.co` is also required for this environment;
the previously exported `hf-mirror.com` endpoint caused Hugging Face metadata
and redirect failures.

### Runtime verification

Using `Qwen/Qwen2.5-0.5B-Instruct`, vLLM started its V1 engine with
FlashAttention, loaded model weights, created 1,044,944 GPU KV-cache token
slots for a 512-token maximum sequence length, and served at port 18000.
Actual calls all returned HTTP 200:

- Single completion: `1+1=` generated `2` as its first token.
- Completion API: a short arithmetic prompt returned generated tokens.
- Batched completion API: two prompts produced two independent choices.
- Chat completion API: the instruction to return `ready` returned exactly
  `ready`.

The smoke server used `--enforce-eager`, so observed throughput is functional
evidence only and is not a paper-performance result. FlashInfer was absent,
causing vLLM to use PyTorch-native top-k/top-p sampling; this does not affect
FlashAttention execution or token/KV-cache semantics. XFormers is not needed
for the validated vLLM V1 FlashAttention path. The test server was stopped
after verification to free GPU memory for PRM/TTS work.

### Remote RTX 5090 discovery

The Windows SSH configuration supplied the 5090 host as `10.129.166.78` with
user `rn_xu29`. The same `Host 5090` alias and forwarding settings are now in
the WSL SSH config. Batch commands should use
`ssh -o ClearAllForwardings=yes 5090 ...` because the interactive forwarding
port 1455 may already be occupied locally.

The remote host has four RTX 5090 GPUs:

- Driver: `595.58.03`
- Driver-reported CUDA capability: `13.2`
- Compute capability: `12.0`
- Memory: `32607 MiB` per GPU
- `nvcc`: not installed

The 13.2 driver can run a CUDA 12.8-built local wheel, but source builds on
the remote host require installation of a CUDA Toolkit first. The remote
environment must still pass an actual vLLM generation test before running
TTS traces.

### Next verification

1. Freeze the local working vLLM artifact source and SHA256, then encode the
   CUDA 12.8 and loopback requirements in the uv deployment scripts.
2. Install an appropriate CUDA Toolkit on the 5090 host or transfer the
   verified compatible wheel, then run the same API smoke test there.
3. Start the TTS policy worker with the loopback variables and run the
   beam-2, branch-4 AIME smoke configuration before adding trace collection.

No `git add` or `git commit` has been performed.
