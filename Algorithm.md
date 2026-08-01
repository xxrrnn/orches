# Trace

## TTS

Compute-Optimal-TTS 的 baseline beam search 已插入可复现 trace。它记录真实的
policy 生成、Skywork PRM reward、beam selection 与最终正确性；不包含 Duplex、
AttAcc 或 ORCHES 的调度优化。这个 trace 可作为修改后的 Duplex/AttAcc 的输入。

### Reproducibility

trace run 默认启用严格确定性模式（`TTS_STRICT_DETERMINISM=1`）。必须固定
`TTS_SEED`、模型 revision、prompt、解码参数和搜索参数；每个 run 必须使用新的
`TTS_RUN_ID`。seed 会随 `LMCallingConfig` 进入 Ray actor，再从
`seed + policy model + prompt` 派生为请求 seed；vLLM worker 同时以
`ORCHES_TTS_SEED` 设置 engine seed，避免重启时使用随机的 engine 初始状态。

为避免 GPU kernel 与并行随机数消费造成差异，严格模式关闭 TF32 和非确定性
cuDNN 行为，policy 将 `n > 1` 候选按固定候选序号串行生成，PRM 使用 eager
attention 且 policy/PRM concurrency 均为 1。每次 `run` 先重复调用固定 fixture
验证 policy token 序列和 PRM reward；指纹不一致会失败，不会写入新的 trace。
可单独执行：

```bash
TTS_SEED=0 TTS_PRM_GPU=0 TTS_POLICY_GPU_MEMORY_UTILIZATION=0.50 \
  bash scripts/4_generate_qwen05_skywork_trace.sh verify
```

这优先保证同一软件/驱动/GPU 配置下的可比较 trace，而非吞吐。串行多候选、单并发
和 eager PRM 会明显降低速度；模型、prompt、temperature、top-p/top-k、token
上限、beam 和 width 本身不变。由于并行 `n` 与串行 `n=1` 的随机数消费不同，严格
模式的候选集合可能不同于旧的非严格 trace；比较实验必须全部使用同一严格模式。

在新机器（例如 5090）上，先运行一次 `start`，以不同 `TTS_RUN_ID` 连续生成两份
trace；停止并重新 `start` 后再生成一份。比较时忽略 manifest 的 `run_id`，并确认
`manifest`、`hw/problem_0000.json`、`sw/problem_0000.json` 的规范化 JSON 一致。
不同 GPU、CUDA、PyTorch、vLLM 或模型 revision 之间不应在未验证前假定 byte-for-byte
一致；manifest 记录 source revision 与 patch series，方便识别这种环境差异。

默认 TTS runner 使用已经在 RTX 5090 GPU 0 上验证过的复现配置：

```text
policy: Qwen/Qwen2.5-Math-1.5B-Instruct
reward: Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
task: AIME24 first 3 examples
beam: 1
width: 4
depth: 4
max_new_tokens: 4096
seed: 0
policy GPU: 0
PRM GPU: 0
policy gpu_memory_utilization: 0.35
policy max_model_length: 8192
```

该配置在同一台服务器、同一物理 GPU 0 上做过“模型离开显存后重载”的对比：先生成
`fixture-20260801-run4`，停止 TTS/vLLM 并确认 GPU 0 显存释放到空闲，再用相同参数
重新加载模型并生成 `fixture-20260801-run5`。两次运行的 output 目录完全一致；忽略
`run_id` 后，software trace 和 hardware trace 的规范化 SHA256 也完全一致：

```text
trace aggregate:  2f2ff7a9f44abe955c64094deb401a89420c73abd775df4c138e07d78b579e10
output aggregate: 2ad4dde8567118a8d470b651fe9432436ee9beb043c82430ce05760db10c3004
```

长时间间隔（例如一周后）重跑时，应把“同一物理 GPU”视为复现条件的一部分。模型
离开显存不是问题，只要重启时保留同一软件栈、模型 snapshot、seed、解码参数、搜索
参数和本地服务环境变量。换到另一张 GPU 即使型号相同，也必须先在那张 GPU 上重新跑
两份 trace 并建立自己的 baseline；不要在未实测前假定不同 GPU 的 kernel 调度和数值
路径能 byte-for-byte 一致。

保持默认复现配置时，常规启动与运行命令为：

```bash
bash scripts/3_run_compute_optimal_tts_example.sh start-beam
TTS_RUN_ID=Qwen1.5/Skywork-1.5B/AIME24/b1/w4/seed0/<new-run-id> \
  bash scripts/3_run_compute_optimal_tts_example.sh run-beam
```

如果要跑完整 AIME24，把调度批大小一起改成 30，避免 evaluator 只分配部分题目：

```bash
TTS_RUN_ID=Qwen1.5/Skywork-1.5B/AIME24/b1/w4/seed0/full-aime24 \
TTS_QUESTION_MAX_NUM=0 \
TTS_BATCH_SIZE=30 \
  bash scripts/3_run_compute_optimal_tts_example.sh run-beam
```

`max_new_tokens=4096` 需要 vLLM 以 `TTS_MAX_MODEL_LENGTH=8192` 启动，使 prompt 与
生成上限能同时放入 KV cache。Qwen 1.5B 的模型配置声明原生
`max_position_embeddings=4096`，因此这个设置是为了复现实验而显式放宽上下文；如果
改回 `TTS_MAX_MODEL_LENGTH=4096`，4096-token 生成请求会因为 prompt 占用而不再是
同一个配置。

### 目录与读取顺序

每次运行创建一个全新的目录：

```text
traces/<policy-label>/<reward-label>/<task>/b<beam>/w<width>/
  manifest.json
  hw/problem_<id>.json
  sw/problem_<id>.json
```

`hw` 是 simulator-facing workload，不含题目、prompt、候选文本或 PRM 输入文本；
`sw` 是软件侧的可解释记录，包含这些文本、reward 和最终结果。两份文件用相同的
`branch_id` 关联，但不应互相嵌入。

一次搜索的逻辑时序为：

```text
p-0 policy -> r-0 reward -> s-0 selection -> p-1 -> r-1 -> s-1 -> ...
```

- policy ID 为 `p-<n>`，位于 `policy_generations` / `generation_events`；
- reward ID 为 `r-<n>`，位于 `prm_forwards` / `reward_events`；
- 每个 `r-*` 通过 `generation_id` 指向对应的 `p-*`；
- selection 通过 `parent_branch_id`、`kept_branch_ids` 和 `pruned_branch_ids`
  连接分支。

JSON 顶层按类型存放（所有 policy、所有 reward、所有 selection），不是交替的
全局事件数组。Duplex/AttAcc 必须根据上述 ID 重建 `p -> r -> selection` 时序，不能
按数组位置错误地把全部 policy 放在全部 reward 之前。

### 记录内容

`hw/problem_<id>.json`：

- 每个分支的父分支、`shared_prefix_tokens`、`generated_tokens`、是否被选择；
- policy prefill、动态 decode segment、每段 batch size 与 sequence length；
- PRM 的 `valid_input_tokens` 与真实 execution shape。Skywork 是串行评分，故每个
  candidate 对应一个 `[1, L]` execution unit；
- beam selection 与最终保留分支。

`sw/problem_<id>.json`：

- policy prompt、候选文本与横向排列的 `output_token_text`；
- policy candidate 的 `token_length`，等于硬件 trace 中同一 `branch_id` 的
  `generated_tokens`；
- reward candidate 的 `token_length`，等于同一 `r-*` hardware record 中该分支的
  `valid_input_tokens`；
- `reward_score`（与兼容字段 `prm_score` 相同）和全量 `step_rewards`；
- `selected`、`final_selected` 与 `result.is_correct`。

`tree_max_width` 是一次扩展的上限，不保证每次实际得到同样多候选：重复候选会被
去重，已终止分支也不会继续扩展。因此 consumer 应以文件中的实际 branch/event 数量
作为 workload，而非假定始终等于 width。

### 插桩位置

插桩实现保存在 `patches/compute-optimal-tts/`，并由
`scripts/1_env_compute_optimal_tts.sh` 按顺序应用：

- `002-tts-baseline-tracing.patch`：建立 manifest、trace recorder、policy/reward/
  selection/结果记录，以及 token 与 PRM execution shape；
- `003-vllm-multisample-aggregation.patch`：聚合 vLLM 0.9 的异步多候选输出，确保
  `n > 1` 不丢候选；
- `004-compact-output-token-text.patch`：使 `output_token_text` 保持单行；
- `005-reward-score-alias.patch`：增加显式 `reward_score`；
- `006-software-candidate-token-lengths.patch`：在 software candidate 中写入与硬件
  trace 对齐的 token length。

核心 hook 在上游 checkout 的 `reason/tracing/tts_trace.py`，由
`reason/evaluation/methods.py` 创建 recorder，`reason/guided_search/tree.py` 在每次
扩展、PRM 调用和选择时写入事件。

### 运行 Qwen 0.5B + Skywork 1.5B 示例

首次准备上游源码、`environments/tts` 和模型：

```bash
bash scripts/0_setup.sh
bash scripts/1_env_compute_optimal_tts.sh
```

此机器只有一张 16GB GPU，policy 与 PRM 共用 GPU 0，policy 显存上限设为 50%。
先启动服务，等待两个模型都出现在 status 的 `Controller models` 中：

```bash
TTS_PRM_GPU=0 TTS_POLICY_GPU_MEMORY_UTILIZATION=0.50 \
  bash scripts/4_generate_qwen05_skywork_trace.sh start

TTS_PRM_GPU=0 TTS_POLICY_GPU_MEMORY_UTILIZATION=0.50 \
  bash scripts/4_generate_qwen05_skywork_trace.sh status
```

运行单题 AIME24 示例：

```bash
TTS_PRM_GPU=0 TTS_POLICY_GPU_MEMORY_UTILIZATION=0.50 \
  bash scripts/4_generate_qwen05_skywork_trace.sh run
```

默认参数为 Qwen 0.5B、Skywork 1.5B、AIME24、beam=2、width=4、depth=4、1 道题和
64 个新 token，输出：

```text
traces/Qwen0.5/Skywork-1.5B/AIME24/b2/w4/
```

配置其他模型对或搜索参数：

```bash
TTS_POLICY_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
TTS_PRM_MODEL=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
TTS_POLICY_LABEL=Qwen0.5 \
TTS_PRM_LABEL=Skywork-1.5B \
TTS_BEAM_SIZE=4 \
TTS_TREE_MAX_WIDTH=8 \
TTS_TREE_MAX_DEPTH=5 \
TTS_QUESTION_MAX_NUM=3 \
TTS_MAX_NEW_TOKENS=256 \
TTS_PRM_GPU=0 \
TTS_POLICY_GPU_MEMORY_UTILIZATION=0.50 \
  bash scripts/4_generate_qwen05_skywork_trace.sh run
```

width 必须能被 beam 整除。每个 run ID 的 result/trace 目录只能创建一次；需要重跑时
改变 label、task、beam/width 或显式设置一个新的 `TTS_RUN_ID`。当 `n > 1` 时不要设
`TTS_TEMPERATURE=0`，因为当前 vLLM 会拒绝 greedy multi-sampling。最后释放 GPU：

```bash
bash scripts/4_generate_qwen05_skywork_trace.sh stop
```
