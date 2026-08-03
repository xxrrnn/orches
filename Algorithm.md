# Trace

## TTS

Compute-Optimal-TTS 的 baseline beam search 已插入可复现 trace。它记录真实的
policy 生成、Skywork PRM reward、beam selection 与最终正确性；不包含 Duplex、
AttAcc 或 ORCHES 的调度优化。这个 trace 可作为修改后的 Duplex/AttAcc 的输入。

## ORCHES 论文复现实验矩阵

本节只记录可以从 [paper/orches.pdf](/home/xrn/projects/pim/orches/paper/orches.pdf)
直接获得的实验信息。论文把 text pipeline 归因于 [18]、vision pipeline 归因于
[36]；未在论文中给出而必须从原算法/代码补齐的字段集中列在最后。这里的“宽度”是
search-tree width/branch count，应映射到 trace 的 `tree_max_width`，不是 beam size。

### 系统、模拟器和比较对象

| 项目 | 论文明确设置 | 复现含义 |
| --- | --- | --- |
| GPU baseline | NVIDIA AGX Orin | 作为 GPU-normalized speedup 与能效的分母。 |
| PIM | 每 bank 16 个 multiplier/adder；总容量 32 GB；2048 banks | PIM 总容量由 edge 约束缩小，但仍满足全部 benchmark 的内存需求。 |
| 片外带宽 | 204.8 GB/s，与 AGX Orin 匹配 | 作为 simulator 的 off-chip bandwidth。 |
| SoC 带宽 sweep | 可用 SoC memory bandwidth 的 100%、75%、50% | 图 11 和文本实验必须分别跑三档，而不是只跑满带宽。 |
| Simulator | 扩展开源 AttAcc；使用修改版 Ramulator2；前端负责 task scheduling、后端负责 PIM memory simulation | AttAcc 是 ORCHES simulator 的基础；Duplex 是比较的 baseline。 |
| 比较对象 | standalone GPU、AttAcc、Duplex、ORCHES | 不同技术的 PIM mapping、prediction 与 memory structuring 开关必须隔离。 |
| unit latency/energy | 延续先前 AttAcc 工作已验证的 GPU/PIM unit latency 与 energy | 系统能耗由计数器统计的数据移动量和各类计算数乘以 unit energy 得到。 |

### 工作负载与模型矩阵

| 实验/论文位置 | Policy（生成） | Reward / PRM（验证） | Benchmark | 搜索与宽度 | 论文明确的额外条件 | 最小复现实验 |
| --- | --- | --- | --- | --- | --- | --- |
| Text 主实验，图 11、表 1、表 5 | `Llama3.2-1B`、`Qwen2.5-1.5B`、`Qwen2.5-3B` | `Qwen2.5-1.5B-PRM-Tuned`、`Qwen2.5-7B-PRM-Tuned`、`Llama3.1-8B-PRM-Tuned` | MATH500 | Beam search；全部 `3 x 3 = 9` model pairs；branch count `2..8` | policy/PRM 组合在 [18] 中的 generation quality 可与或超过 Llama3.1-405B；对每个配置扫 100/75/50% SoC bandwidth | `9 x 7 x 3 = 189` 个 trace/replay 配置，再对每个配置运行 GPU、AttAcc、Duplex、ORCHES。 |
| Text 泛化，表 2 | 论文说明沿用 text TTC pipeline，但未逐项指定 policy | 未逐项指定 PRM；表的列是 Qwen2.5-1.5B、Qwen2.5-7B、Llama3.1-8B | LiveCodeBench | 表中 width `2`、`4` | 扫 100/75/50% bandwidth | 在未取得原始配置前，先使用主实验 3x3 矩阵；不要猜测某个 policy/PRM 子集。 |
| Vision 主实验，表 3 | Fine-tuned `Llama-3.2-11B-Vision-Instruct` | 同一 fine-tuned `Llama-3.2-11B-Vision-Instruct` | MATHVista | TTC vision pipeline；width `2`、`4` | 论文称该 pipeline 超过 GPT-4o-mini、Llama-3.2-90B-Vision-Instruct 等较大模型 | 每个 width 采集/重放一个 policy=PRM 的 vision trace，并按 short/medium/long question 分组。 |
| T1 assignment 消融，图 12、5.3 节 | `Qwen2.5-3B` | `Qwen2.5-7B-PRM-Tuned` | MATH500 | 随 search width 变化 | 关闭 T2；比 GPU、AttAcc、ORCHES-A、B、C | 运行 A/B/C 的同一组 trace，避免以不同 branch 结果比较 assignment。 |
| T2 case study，图 13 | `Llama3.2-1B`，或 `Qwen2.5-3B` | `Llama3.1-8B-PRM-Tuned` | MATH500 | 文本 beam-search pipeline；宽度未单列 | small PRM 是原 8B PRM 的前 10 层；large PRM 是其余层；示例预测 50 或 52 tokens，示例并行预执行 small PRM 的 60% | 记录 prediction、speculative work、squash、实际 overlap 和每段 token 数。 |
| T1/T2 组合消融，表 6、5.6 节 | `Qwen2.5-3B` | 三种：Qwen2.5-1.5B、Qwen2.5-7B、Llama3.1-8B PRM | MATH500 | question difficulty 平均；宽度未单列 | 比 T1 only、T2 only、T1+T2 | 对每个 PRM 跑三个 feature 配置，且维持同一模型、trace 和硬件。 |
| T3 memory structuring，表 5、5.5 节 | Llama3.2-1B、Qwen2.5-1.5B、Qwen2.5-3B | Qwen2.5-1.5B、Qwen2.5-7B、Llama3.1-8B PRM | MATH500 | 与 text 主实验相同的设置 | 仅 selected branch 继续执行；每 3--5 个 reasoning step/PRM verification 重整一次 | 记录 context-KV footprint、碎片、搬移字节、buffer 面积与 runtime overhead。 |

### Figure 11 的 baseline 标注

| 图 11 列 | Policy 行 | PRM 标注 | 带宽 | 比较指标 |
| --- | --- | --- | --- | --- |
| ORCHES | Llama3.2-1B、Qwen2.5-1.5B、Qwen2.5-3B | Qwen2.5-1.5B | 100%、75%、50% | 相对 AGX Orin GPU 的 normalized speedup。 |
| Baseline-AttAcc | 同上三种 policy | Qwen2.5-7B | 100%、75%、50% | 相对 GPU 的 normalized speedup。 |
| Baseline-Duplex | 同上三种 policy | Llama3.1-8B | 100%、75%、50% | 相对 GPU 的 normalized speedup。 |

论文图 11 为不同 PRM 尺寸的系统列分别标注 PRM，不应把它误读为严格同一 model pair
下的 AttAcc-vs-Duplex 对比。复现报告必须保留每列实际 policy、PRM 与带宽。

### 论文报告的数值结果

| 指标 | 论文数值 |
| --- | --- |
| Text 平均 speedup | ORCHES 相对 GPU 平均 `4.16x`。 |
| Text 平均能效 | ORCHES 相对 GPU 平均 `2.45x`；表 1 数值如下。 |
| LiveCodeBench 平均 speedup | ORCHES 相对 GPU 平均 `4.24x`；表 2 数值如下。 |
| Vision 平均 speedup | ORCHES 相对 GPU 平均 `3.10x`；short/medium/long 的范围为 `2.32x..4.85x`。 |
| T1 assignment 消融 | ORCHES 平均相对 GPU `3x`、相对 AttAcc `1.5x`；AttAcc 本身相对 GPU `2x`。 |
| T2 predictor | 历史对齐前平均 prediction accuracy 约 `52%`，后约 `78%`。 |
| T3 memory | context-memory footprint 平均节省 `65%`；新增 buffer area `12%`；平均 runtime overhead `0.12%`。 |
| T1/T2 utilization | GPU：T1 `97.9%`、T2 `62.2%`、T1+T2 `93.21%`；PIM：`43.6%`、`66.7%`、`61.0%`。 |

#### 表 1：相对 GPU 的能效（MATH500，平均于 width 与 question length）

| PRM \ Policy | Llama3.2-1B | Qwen2.5-1.5B | Qwen2.5-3B |
| --- | ---: | ---: | ---: |
| Qwen2.5-1.5B | 1.96x | 2.07x | 1.87x |
| Qwen2.5-7B | 3.23x | 2.57x | 2.14x |
| Llama3.1-8B | 3.40x | 2.71x | 2.13x |

#### 表 2：相对 GPU 的 LiveCodeBench speedup

| 可用 SoC 带宽 \ PRM | Qwen2.5-1.5B | Qwen2.5-7B | Llama3.1-8B |
| --- | ---: | ---: | ---: |
| 100% | 3.85x | 3.19x | 2.73x |
| 75% | 4.98x | 3.77x | 3.27x |
| 50% | 6.93x | 5.10x | 4.31x |

#### 表 3：相对 GPU 的 MATHVista speedup

| Search width | Short QA | Medium QA | Long QA |
| --- | ---: | ---: | ---: |
| 2 | 3.26x | 3.35x | 4.85x |
| 4 | 2.47x | 2.35x | 2.32x |

#### 表 4：history alignment 前后 branch-prediction accuracy（MATH500）

| 难度 | Llama3.2-1B policy | Qwen2.5-1.5B policy | Qwen2.5-3B policy |
| --- | ---: | ---: | ---: |
| Level 1 | 51.4% -> 73.3% | 56.1% -> 82.4% | 61.1% -> 79.5% |
| Level 2 | 50.7% -> 80.1% | 56.8% -> 82.6% | 61.5% -> 79.2% |
| Level 3 | 53.2% -> 82.2% | 57.5% -> 82.8% | 59.9% -> 79.5% |
| Level 4 | 52.7% -> 82.3% | 57.7% -> 83.1% | 59.7% -> 79.6% |
| Level 5 | 52.6% -> 83.0% | 57.9% -> 83.1% | 59.8% -> 80.3% |

#### 表 5：T3 context-memory footprint saving（MATH500）

| PRM \ Policy | Llama3.2-1B | Qwen2.5-1.5B | Qwen2.5-3B |
| --- | ---: | ---: | ---: |
| Qwen2.5-1.5B | 63% | 68% | 67% |
| Qwen2.5-7B | 64% | 71% | 65% |
| Llama3.1-8B | 66% | 78% | 65% |

#### 表 6：T1/T2 相对 GPU speedup（MATH500，policy Qwen2.5-3B）

| 配置 \ PRM | Qwen2.5-1.5B | Qwen2.5-7B | Llama3.1-8B |
| --- | ---: | ---: | ---: |
| T1 only | 4.1x | 2.9x | 3.1x |
| T2 only | 3.1x | 2.8x | 2.9x |
| T1 + T2 | 4.4x | 3.2x | 3.4x |

### TTS trace 必须记录的论文动态性

| 论文机制 | 可直接得到的运行规则 | trace/simulator 必须记录 |
| --- | --- | --- |
| Variable parallelism | policy 是 token-by-token decoding；PRM 主要是 prefill；shared 和 unique KV 同时存在，且 shared:unique 比例随 search depth 变化 | 每 token active branch batch、每 branch context length、policy/PRM 事件依赖、shared/private KV 字节。 |
| T1 offline assignment | 小 batch：linear 与全部 attention 放 PIM；中 batch：GPU 跑 shared-KV query、PIM 跑 linear 和 unique-KV query；大 batch：GPU 跑 linear 与 shared attention、PIM 跑 unique-KV query | 每 event 的 batch、shared/unique KV 分解、GPU/PIM/通信时间与 assignment。 |
| T1 online compensation | shared KV 会随 step 累积；在每个 reasoning step 计算并动态调整 GPU/PIM 分割 | 每步分割比例、队列时间、数据传输；论文称 transfer 平均约占总 runtime 的 8.3%。 |
| T2 prediction/pipeline | small PRM 预测下一步会被保留的 branch；历史 alignment 用 large PRM 历史 score 替换 small PRM 历史 score；可与 generation 重叠 verification | predictor score、预测 branch、large-PRM 最终选择、speculation 的已执行 token、squash 与 overlap。 |
| T3 memory structuring | 剪枝后只保留 selected branches；将 isolated context data 合并以恢复连续存储；每 3--5 step 重整 | parent-child KV page ownership、refcount、prune、物理地址、碎片、compaction bytes/time/energy。 |

### 论文没有披露、不得猜测的参数

| 参数 | 论文状态 | 复现要求 |
| --- | --- | --- |
| Text beam size | 仅说明 beam search，未给独立数值 | 从 [18] 或作者代码确认，写入 `TTS_BEAM_SIZE` 和 manifest。 |
| Text search depth | 未给独立数值 | 从 [18] 或作者代码确认，写入 `TTS_TREE_MAX_DEPTH`。 |
| 每个 benchmark 的 prompt、temperature、top-p/top-k、max tokens、seed | 未给 | 与原 pipeline 同步；全部冻结并纳入 trace。 |
| LiveCodeBench 使用的具体 policy | 表 2 未列出 | 在报告前从原始配置确认；当前不能由表 2 的 PRM 列反推。 |
| MATHVista 的独立 beam/depth、prompt 与 decoding 参数 | 未给 | 从 [36] 或作者代码确认。 |
| 详细 AGX Orin compute/memory clock、PIM timing/energy table | 论文只给出上述摘要与“沿用 prior work” | 从 AttAcc/ORCHES 实现和引用 [25] 补齐，不能以本机 A100/H100 参数代替。 |

当前仓库的 `Qwen2.5-Math-1.5B-Instruct + Skywork-1.5B` AIME24 trace 仅是 adapter
smoke test；其 `beam=1`、`width=4`、`depth=4` 不属于论文配置，不能替代上表中未披露
的 beam/depth。

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
  bash scripts/test/4_generate_qwen05_skywork_trace.sh verify
```

这优先保证同一软件/驱动/GPU 配置下的可比较 trace，而非吞吐。串行多候选、单并发
和 eager PRM 会明显降低速度；模型、prompt、temperature、top-p/top-k、token
上限、beam 和 width 本身不变。由于并行 `n` 与串行 `n=1` 的随机数消费不同，严格
模式的候选集合可能不同于旧的非严格 trace；比较实验必须全部使用同一严格模式。

在新机器（例如 5090）上，先运行一次 `start`，以不同 `TTS_RUN_ID` 连续生成两份
trace；停止并重新 `start` 后再生成一份。比较时忽略 manifest 的 `run_id`，并确认
`manifest`、`hw/problem_0000.json`、`sw/problem_0000.json` 的规范化 JSON 一致。
不同 GPU、CUDA、PyTorch、vLLM 或模型 revision 之间不应在未验证前假定 byte-for-byte
一致；manifest 记录 server alias、SSH URL、policy/PRM GPU index、GPU UUID、GPU
name、PCI bus id、source revision 与 patch series，方便识别这种环境差异。

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
  bash scripts/test/4_generate_qwen05_skywork_trace.sh start

TTS_PRM_GPU=0 TTS_POLICY_GPU_MEMORY_UTILIZATION=0.50 \
  bash scripts/test/4_generate_qwen05_skywork_trace.sh status
```

运行单题 AIME24 示例：

```bash
TTS_PRM_GPU=0 TTS_POLICY_GPU_MEMORY_UTILIZATION=0.50 \
  bash scripts/test/4_generate_qwen05_skywork_trace.sh run
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
  bash scripts/test/4_generate_qwen05_skywork_trace.sh run
```

width 必须能被 beam 整除。每个 run ID 的 result/trace 目录只能创建一次；需要重跑时
改变 label、task、beam/width 或显式设置一个新的 `TTS_RUN_ID`。当 `n > 1` 时不要设
`TTS_TEMPERATURE=0`，因为当前 vLLM 会拒绝 greedy multi-sampling。最后释放 GPU：

```bash
bash scripts/test/4_generate_qwen05_skywork_trace.sh stop
```
