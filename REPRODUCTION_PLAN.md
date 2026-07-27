# ORCHES 复现与 Baseline 实施计划

## 1. 目标与复现边界

本项目的目标是在 `orches/` 内建立一个可审计、可重复运行的 ORCHES baseline，复现论文
[ORCHES: Orchestrated Test-Time-Compute-based LLM Reasoning on Collaborative GPU-PIM Heterogeneous System](<./Li et al. - 2025 - ORCHES Orchestrated Test-Time-Compute-based LLM Reasoning on Collaborative GPU-PIM HEterogeneous Sy.pdf>)
中的以下内容：

1. TTC reasoning workload：policy model 生成候选，PRM 验证并剪枝，逐 step 推进。
2. ORCHES 硬件：AGX Orin 类 host GPU、controller die、32 GB bank-level PIM。
3. Technique 1A：基于 roofline/延迟模型的离线 GPU-PIM 分配。
4. Technique 1B：随 shared KV 增长而动态调整的在线协同补偿。
5. Technique 2A：带 history alignment 的 candidate verification predictor。
6. Technique 2B：利用 GPU 空闲窗口执行的 pipelined verification。
7. Technique 3：address cache、KV buffer 与动态 memory reorganization。
8. 论文第 5 节中的系统评测、消融、能效、利用率和内存占用结果。

复现分成两个互相解耦的平面：

```text
真实模型/数据集运行
    -> 记录候选长度、分支选择、PRM 分数、KV 长度、token 时间线
    -> 确定性 workload trace
    -> 系统级离散事件模拟器
       -> GPU analytical/calibrated model
       -> GPU-PIM 调度、推测执行、rollback、同步
       -> Ramulator2 PIM command trace
    -> latency / energy / utilization / memory footprint
    -> 论文图表
```

真实模型运行负责保持原 reasoning pipeline 的准确率；性能模拟器只回放已经记录的控制流，
不生成或篡改模型答案。这样 ORCHES 的推测执行在预测错误时 rollback，不会改变最终输出，
对应论文“without any loss in accuracy”的设定。

## 2. 证据等级和复现纪律

每个配置字段都必须带来源，使用以下等级：

| 标记 | 含义 | 使用规则 |
|---|---|---|
| `PAPER` | 论文正文、公式、图或表明确给出 | 默认主实验配置 |
| `INHERITED` | 论文声明沿用 AttAcc，值来自固定 commit | 记录上游文件和 commit |
| `CALIBRATED` | 从 AGX Orin 或已验证实现测量得到 | 保存原始测量、拟合误差和环境；作为独立校准 profile |
| `ASSUMED` | 论文未披露，只能工程补全 | 不得静默使用；必须做敏感性分析 |

不得为了拟合论文最终 speedup 而反向调参。主结果分为 `paper_method` 和
`orin_calibrated` 两个 profile：前者遵循 Sec. 5.1，使用 AttAcc 风格解析模型和继承单位开销；
后者在目标硬件可用时增加独立校准。两者不得混合参数。完整 trace 采集方案见
`docs/trace-collection-plan.md`。

## 3. 论文已明确的配置

### 3.1 硬件

| 参数 | 值 | 来源 |
|---|---:|---|
| GPU baseline | NVIDIA Jetson AGX Orin 32 GB | Sec. 5.1 |
| GPU/PIM 可见带宽 | 204.8 GB/s | Sec. 5.1 |
| PIM 总容量 | 32 GB | Sec. 5.1 |
| PIM memory banks | 2048 | Sec. 5.1 |
| 每 bank 计算单元 | 16 multipliers + adders | Sec. 4.1, 5.1 |
| PIM 计算位置 | memory bank 内 GEMV units | Sec. 4.1 |
| controller die | address cache、state machine、accum、softmax、shared-KV buffer | Fig. 7, Sec. 4.1 |
| memory timing/单位能耗 | 继承 AttAcc | Sec. 5.1 |
| SoC 带宽扫描 | 100%、75%、50% | Fig. 11, Sec. 5.1 |

Ramulator2 中计划保留 AttAcc 的 `HBM3_8Gb_2R` preset，并将其 16 channels 扩展为
32 channels。每 channel 含 2 pseudochannels、每 pseudochannel 含 2 ranks、每 rank 含
4 bank groups、每 bank group 含 4 banks，因此得到 `32*2*2*4*4=2048` banks；
`32 channels * 8 Gibit/channel / 8=32 GiB`。channel 扩展是根据论文容量与 bank 数反推的
`ASSUMED` 映射，正式报告必须保留敏感性分析。
HBM 内部 bank 带宽与 host/controller I/O 的 204.8 GB/s 必须分别建模，分别对应论文中的
`BW_PIM` 和 `BW_PIM_IO`。

### 3.2 文本任务

| 维度 | 配置 |
|---|---|
| 数据集 | MATH500；泛化实验增加 LiveCodeBench |
| TTC pipeline | 论文引用 [18] 的 compute-optimal TTS |
| policy models | Llama3.2-1B、Qwen2.5-1.5B、Qwen2.5-3B |
| PRM models | Qwen2.5-1.5B-PRM-Tuned、Qwen2.5-7B-PRM-Tuned、Llama3.1-8B-PRM-Tuned |
| model combinations | 3 x 3 = 9 |
| search width | 2 到 8；完整扫描为所有整数 `{2, 3, 4, 5, 6, 7, 8}`，绘图时按论文坐标取子集 |
| question difficulty | Level 1 到 Level 5，沿用 [18] 的分级数据 |

### 3.3 视觉任务

| 维度 | 配置 |
|---|---|
| 数据集 | MathVista |
| pipeline | 论文引用 [36] 的 LLaVA-o1/LLaVA-CoT |
| policy/PRM | fine-tuned Llama3.2-11B-Vision-Instruct |
| search width | 2、4 |
| question length | short、medium、long；论文未给阈值，需保留为 `ASSUMED` 并报告分桶规则 |

### 3.4 论文结果只作为验证锚点

| 指标 | 论文值 |
|---|---:|
| 文本任务平均 speedup vs. GPU | 4.16x |
| 文本任务平均 energy efficiency vs. GPU | 2.45x |
| LiveCodeBench 平均 speedup vs. GPU | 4.24x |
| 视觉任务平均 speedup vs. GPU | 3.10x |
| 视觉任务 speedup 范围 | 2.32x - 4.85x |
| Technique 1 平均 speedup vs. GPU / AttAcc | 3.0x / 1.5x |
| history alignment 前后 predictor accuracy | 约 52% -> 约 78% |
| Technique 3 平均 context memory saving | 65% |
| Technique 3 buffer area overhead | 12% |
| Technique 3 runtime overhead | 0.12% |
| GPU utilization: T1 / T2 / T1+T2 | 97.9% / 62.2% / 93.21% |
| PIM utilization: T1 / T2 / T1+T2 | 43.6% / 66.7% / 61.0% |
| GPU-PIM data transfer 占总时间 | 约 8.3% |

这些数值用于发现实现错误和解释偏差，不作为调参目标。

### 3.5 必须逐式实现的调度模型

Technique 1A 的单层模型为：

```math
T_{GPU} = \frac{WD^2}{CC_{GPU}} + \frac{2WD + D^2}{BW_{GPU}}
```

```math
T_{PIM} = \frac{WD^2}{CC_{PIM}}
        + \frac{2WD}{BW_{PIM\_IO}}
        + \frac{D^2}{BW_{PIM}}
```

当输出维度中比例 `alpha` 交给 GPU 时：

```math
T_{PIM}(\alpha) = \frac{WD^2(1-\alpha)}{CC_{PIM}}
                + \frac{WD(2-\alpha)}{BW_{PIM\_IO}}
                + \frac{D^2}{BW_{PIM}}
```

```math
T_{GPU}(\alpha) = \frac{WD^2\alpha}{CC_{GPU}}
                + \frac{WD(1+\alpha)+D^2\alpha}{BW_{GPU}}
```

Technique 1B 对多个 KV fragment 使用：

```math
T_{PIM}(\{\alpha_i\}) =
  \sum_i \frac{W_iL_iD(1-\alpha_i)}{CC_{PIM}}
  + \sum_i \frac{L_iD}{BW_{PIM}}
  + \sum_i \frac{W_iD + W_iL_i(1-\alpha_i)}{BW_{PIM\_IO}}
```

```math
T_{GPU}(\{\alpha_i\}) =
  \sum_i \frac{W_iL_iD\alpha_i}{CC_{GPU}}
  + \sum_i \frac{W_iD + W_iL_i\alpha_i}{BW_{GPU}}
```

这些式子按论文的 operation 计数约定实现，不自行补乘 2。详细 Transformer 模型可以使用标准
FLOP 定义，但其 `CC` 必须采用同一计数约定校准，禁止在 scheduler 和 device model 间混用单位。

## 4. 计划中的目录和交付物

```text
orches/
├── README.md
├── REPRODUCTION_PLAN.md
├── pyproject.toml
├── CMakeLists.txt
├── third_party.lock
├── configs/
│   ├── hardware/{agx_orin_32gb,orches_pim_32gb}.yaml
│   ├── models/{policy,prm,vision}/*.yaml
│   ├── workloads/{math500,livecodebench,mathvista}.yaml
│   └── experiments/{figure11,table1,table2,table3,figure12,table4,
│                    figure13,table5,table6}.yaml
├── src/orches/
│   ├── workload/{schema,collector,replay}.py
│   ├── models/{transformer,operators,profiles}.py
│   ├── scheduler/{roofline,offline,online}.py
│   ├── predictor/{scores,history,speculation,pipeline}.py
│   ├── memory/{allocator,address_cache,kv_buffer,compactor}.py
│   ├── sim/{event,resource,gpu,pim,system,energy}.py
│   ├── baselines/{gpu,attacc,duplex}.py
│   └── eval/{matrix,runner,aggregate,plots}.py
├── csrc/orches_ramulator/
│   ├── orches_trace.cpp
│   ├── orches_controller.cpp
│   ├── orches_scheduler.cpp
│   ├── orches_mapper.cpp
│   └── orches_hbm3.cpp
├── patches/ramulator2/
├── scripts/{bootstrap,collect_traces,run_exp,reproduce_all}.sh
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── microbench/
│   └── golden/
├── docs/
│   ├── hardware-contract.md
│   ├── paper-to-code.md
│   ├── assumptions.md
│   ├── calibration.md
│   └── validation.md
└── artifacts/                 # 原始 traces/results，不提交大型数据
    ├── manifests/
    ├── traces/
    ├── raw/
    ├── processed/
    └── figures/
```

`attacc_simulator/`、`Duplex/` 和 `pimba/` 保持不变。ORCHES 使用固定 commit 的上游代码、
本地 patch 和 baseline adapter，避免污染已有 baseline。

## 5. 分阶段实施步骤

### Phase 0：冻结来源和可复现环境

实现：

1. 记录论文 DOI、PDF SHA256、AttAcc commit、Ramulator2 commit、Duplex commit、
   compute-optimal-TTS commit 和 LLaVA-o1 commit。
2. 创建 `third_party.lock`，记录 URL、commit、license、patch SHA256。
3. 建立 Python lockfile、CMake preset、容器/Conda 环境说明和 `scripts/bootstrap.sh`。
4. 每次实验生成 manifest：git commit、dirty state、主机、编译器、配置 hash、trace hash、seed。

论文对应：Sec. 5.1 的 simulator setup 和两个算法 pipeline。

完成标准：干净环境中一条命令可以构建；同一配置两次运行的原始结果 hash 一致。

当前环境差距：`cmake` 和 `clang++` 未安装，AttAcc/Duplex 的 Ramulator2 submodule 未初始化；
当前会话无法访问 GPU。性能模拟器和 `paper_method` 解析 profile 的开发可继续；真实模型 trace
需要在可见 GPU 上采集，`orin_calibrated` profile 需要另行在 AGX Orin 上测量。

### Phase 1：写硬件合同和配置校验器

实现：

1. 在 `docs/hardware-contract.md` 定义 GPU、controller、host link、bank、GEMV、accum、
   softmax、address cache、KV buffer 的容量、带宽、时钟和并发规则。
2. 在 `configs/hardware/` 写 AGX Orin 和 ORCHES PIM 配置；每个字段包含 `value/source/status`。
3. 配置加载时验证：总容量、bank 数、总 MAC 数、单位、带宽层级和地址位宽。
4. 未解决的 `ASSUMED` 字段在主实验启动前必须显式确认，不能使用隐藏默认值。

论文对应：Fig. 7、Sec. 4.1、Sec. 5.1；实现方法遵循根目录 `Tutorial.md` 的硬件合同要求。

测试：配置 schema；`2048 banks` 和 `32 GiB` 推导；错误单位和不一致容量必须失败。

### Phase 2：构建 TTC workload trace

状态：schema v2 已实现 exact token/mask、逻辑 KV block lineage、`search_width` 与
`beam_size` 分离、scalar PRM/pairwise judge 和有序 selection events。真实 pipeline collector
尚未实现，因此目前没有 evaluation trace。

实现：

1. 定义版本化 JSONL schema，每个 request/step 至少记录：
   `dataset_id`、difficulty、policy/PRM、实际提交的 input token IDs/mask、每个 candidate 的
   generated token IDs、逻辑 token-ready 顺序、small/large PRM 分数或 pairwise verifier calls、
   有序 selected/pruned 集合、实际 KV materialization 和 parent/block lineage。所有长度由这些
   token/KV 记录推导，禁止使用输出字符数或重新 tokenize 的文本。collector 的绝对 GPU wall
   time 不进入 Orin replay。
2. 在固定模型、tokenizer、seed、采样参数下运行文本 [18] pipeline，采集 MATH500 和
   LiveCodeBench trace。
3. 运行视觉 [36] pipeline，额外记录 image token 数和 question-length bucket。
4. 将模型输出正确性与 timing trace 分开保存；去除 prompt 正文等非必要敏感数据。
5. 提供 synthetic trace generator，先用小模型形状验证模拟器，不把 synthetic 结果用于论文结论。

论文对应：Sec. 2.2、Fig. 3、Sec. 3、Sec. 5.1 的 algorithm pipeline and dataset。

测试：schema round-trip；selected branch 必须存在；下一 step 的 parent/KV 必须由真实选择导出；
width 4 非顺序胜者和 `beam_size > 1` 的多存活分支正确；固定 seed 重放控制流一致。

完成标准：任一 request 可从 trace 重建完整 reasoning tree，而不需要再次调用模型。

### Phase 3：实现 Transformer 算子与 AGX Orin baseline

实现：

1. 支持 Llama/Qwen 的层数、hidden size、FFN、MHA/GQA、KV heads 和 FP16 数据量。
2. 将 prefilling、decoding、linear、shared KV query、unique KV query、softmax、normalization
   展开为显式算子 DAG。
3. GPU 时间使用 `max(compute_time, memory_time)` 加显式 kernel/launch/同步开销。
   `paper_method` 使用 Sec. 5.1 所述 AttAcc 风格解析模型、论文带宽和显式继承/假设参数；
   `orin_calibrated` 另行使用 Orin microbenchmark 拟合值。官方 peak 只能标为解析上界。
4. 实现 100%、75%、50% SoC bandwidth profile，不同时改变 compute capability。
5. 建立纯 GPU baseline：generation 和 verification 均在 GPU，严格遵守 step 依赖。

论文对应：Sec. 2.1、3.1、Eq. (1)、Sec. 5.1 的 AGX Orin baseline。

测试：算子 FLOPs/bytes 对手算；batch 增大时 linear arithmetic intensity 单调增；unique KV query
arithmetic intensity 保持论文描述的低值；GPU baseline 不出现跨 step 非法重叠。

### Phase 4：实现 ORCHES PIM 和 Ramulator2 后端

实现：

1. 以 AttAcc 固定 commit 的 HBM3-PIM command/state/timing 为基础，不直接修改 sibling repository。
2. 配置 32 GB、2048 banks、每 bank 16 MAC 单元；保留 refresh、ACT/PRE、row conflict 和
   AttAcc power constraint。
3. 扩展 trace frontend 支持 linear GEMV、shared/unique attention、controller accumulation、
   softmax、barrier 和 reorganization read/write。
4. 扩展 address mapper，使权重、shared KV、unique KV 和 compaction destination 可逆映射到
   channel/pseudochannel/rank/bankgroup/bank/row/column。
5. controller scheduler 保持 barrier 内顺序，并为普通访问、PIM compute 和后台 compaction
   建立明确优先级。
6. 输出 command/event counters：ACT、PRE、RD、WR、MAC、ACCUM、SFM、buffer traffic、
   host traffic、stall、row hit、refresh 和 cycles。

论文对应：Fig. 7、Sec. 4.1、Eq. (2)-(3)、Sec. 5.1 的 modified Ramulator2 backend。

测试：单 bank、全 bank broadcast、bank scaling、row locality、refresh、barrier、地址可逆、
读改写和 writeback microbench。PIM 吞吐不得超过内部 bank bandwidth。

完成标准：单算子 Ramulator cycle 和 analytical lower bound 差异可解释；所有 PIM 数据移动均有计数。

### Phase 5：实现 Technique 1A 离线分配

实现：

1. 在 `scheduler/roofline.py` 逐字实现 Eq. (1)-(4)，所有变量带单位：batch/branch width `W`、
   hidden dimension `D`、compute capability `CC`、内部/外部带宽 `BW`。
2. 按论文实现三档 assignment：
   - small `W`：linear 和全部 attention 在 PIM；
   - medium `W`：shared KV query 在 GPU，linear 和 unique KV query 在 PIM；
   - large `W`：linear/shared attention 在 GPU，unique KV query 在 PIM。
3. 对可协同的 linear 层求解 `T_GPU(alpha) = T_PIM(alpha)`，将输出维度的 `alpha` 部分交给 GPU。
4. 只有在 `T_PIM >= max(T_GPU(alpha), T_PIM(alpha))` 时启用协同，否则选择单设备最优方案。
5. 显式加入输入发送、partial FP16 output 收集和 barrier；记录数据移动比例。

论文对应：Sec. 4.2.1、Fig. 8(a)-(e)、Eq. (1)-(4)。

测试：边界 `alpha=0/1`；解析解与数值穷举一致；分配阈值两侧连续；关闭通信时不产生 traffic；
不得将 unique KV query 分给 GPU 作为默认路径。

完成标准：同一层的选择可输出完整解释，包括各候选方案 latency、alpha 和瓶颈资源。

### Phase 6：实现 Technique 1B 在线补偿

实现：

1. 在每个 reasoning step 前，按每段 KV 的 `(L_i, W_i, D)` 计算 Eq. (5)-(7)。
2. 初始令所有层 `alpha_i=1`；从最小 `W_i` 到最大 `W_i` 将层切换到 PIM (`alpha_i=0`)。
3. 找到 `T_PIM` 与 `T_GPU` 关系翻转的 critical layer；其他层固定为 0/1，只对该层求连续 `alpha_t`。
4. shared KV 随 step 增长时重新调度，unique KV 在新 candidate 开始时归零。
5. 系统模拟器并行推进 GPU/PIM 资源，在汇合点执行同步并记录 imbalance stall。

论文对应：Sec. 3.1.2、Fig. 5、Sec. 4.2.2、Fig. 8(f)、Eq. (5)-(7)。

测试：实现结果与小规模 `alpha_i` 穷举最优解比较；shared KV 增长时 workload 必须迁移；
事件时间线不得违反依赖；关闭 T1B 后退化为 T1A。

### Phase 7：实现 Technique 2A predictor 和推测执行

实现：

1. 将 large PRM 的前 10 层作为 small PRM 配置之一，复现论文 Fig. 13 的无额外模型开销案例。
2. small PRM 对当前候选打分；历史 step 不使用 small PRM 自身旧分数，而替换为已完成的
   large PRM 分数，形成 history alignment。
3. small PRM 完成后，PIM 用预测分支开始 step `N+1` generation；GPU 继续运行 large PRM。
4. 预测匹配时保留推测 token；不匹配时取消错误事件、丢弃错误 KV，并从 large PRM 选择的分支重启。
5. verification 优先；推测 generation 只使用剩余资源。推测窗口内关闭 T1，large PRM 完成后立即恢复 T1。
6. 记录 predictor accuracy、正确/错误推测 token、saved latency、rollback latency 和错误 KV bytes。

论文对应：Sec. 4.3.1、Fig. 9(a)-(c)、Table 4、Fig. 13。

测试：history alignment on/off；强制全对/全错 predictor；rollback 后最终 branch、KV 和输出必须与
无推测 baseline 完全一致；错误事件不能消耗未来时间两次。

### Phase 8：实现 Technique 2B pipelined verification

实现：

1. token 生成后立即形成 PRM prefill chunk，而不是等待全部 candidates 完成。
2. 只有 GPU 空闲且累计 token 达到 `min_prefill_tokens` 时发射 chunk；阈值由 GPU roofline 校准。
3. pre-verification 可被关键 generation/large PRM 工作抢占或延迟，但已完成工作不能重复计费。
4. small PRM pre-verification 与 PIM generation 重叠，最终与剩余 verification 合并。
5. 输出 GPU/PIM busy intervals、idle intervals 和关键路径。

论文对应：Sec. 4.3.2、Fig. 9(d)、Fig. 13(b)。

测试：0%、部分、100% overlap；不足阈值不启动；流水线结果分数与非流水线一致；资源区间不重叠计费。

### Phase 9：实现 Technique 3 memory structuring

实现：

1. allocator 分离 model weights、input/shared KV、candidate unique KV；候选剪枝时释放 unique KV，
   形成真实 holes。
2. controller address cache 保存 `candidate_id -> (start, length, generation)`；命中时模拟一次 SRAM
   lookup 加一次 DRAM access，替代两个有依赖的 DRAM accesses。
3. 计算 `beta = total_memory_holes / memory_for_reasoning`；支持 beta threshold 和固定每 3/4/5 steps
   两种策略，主配置和敏感性结果同时保存。
4. compactor 生成真实 DRAM RD/WR trace，把 live blocks 搬到连续区域；更新 address cache 采用
   generation/version，避免并发读到旧地址。
5. controller shared-KV buffer 保存 QKV accumulation 结果；后台回写到 banks，避免经 host GPU 搬运。
6. 如果 decoding 全在 PIM，模拟 GPU 获取最新 KV；否则利用 controller buffer 快速同步。
7. 记录 peak/live/allocated bytes、fragmentation、bytes moved、compaction stalls、cache hits 和 buffer traffic。

论文对应：Sec. 3.3、Fig. 10、Sec. 4.4、Table 5、Sec. 5.5。

测试：allocate/prune/compact 属性测试；compaction 前后逻辑地址读取一致；并发访问版本正确；
禁用 T3 时 holes 保留；面积、runtime 和 memory saving 分项报告。

### Phase 10：实现公平 baseline

实现：

1. `GPU`：相同 trace、精度、模型和 204.8 GB/s SoC profile，无 PIM。
2. `AttAcc`：固定论文 [25]/本地 `attacc_simulator` commit，attention 放 PIM；对 TTC 的扩展只做
   workload adapter，不加入 ORCHES 的 T1/T2/T3。
3. `Duplex`：固定论文 [40]/本地 `Duplex` commit；使用其 bank/Logic-PIM 配置，通过同一 trace adapter。
4. `ORCHES-A`：所有可支持计算放 PIM，T2 关闭。
5. `ORCHES-B`：启用 T1A adaptive linear assignment，T2 关闭。
6. `ORCHES-C`：在 B 上增加 T1B dynamic compensation，T2 关闭。
7. `ORCHES`：T1 + T2 + T3 完整系统。

论文对应：Sec. 5.1、Fig. 11、Sec. 5.3、Fig. 12。

公平性检查：相同 workload trace、权重/激活精度、容量、GPU 数、带宽、refresh、模型输出；
任何 baseline 特有假设必须进入结果表，而不是埋在 adapter 中。

### Phase 11：能耗、面积和利用率

实现：

1. 事件能耗分别统计 GPU compute/memory、ACT/PRE、RD/WR、PIM MAC、accum、softmax、SRAM cache、
   KV buffer、GPU-PIM link 和 compaction。
2. 单位能耗继承 AttAcc 的项标记 `INHERITED`；新增 controller SRAM/逻辑用 CACTI/综合结果标记
   `CALIBRATED`，不可直接把 energy 命名为 power。
3. `energy_efficiency = baseline_energy / orches_energy`，同时保存绝对 J/request。
4. 利用率按 busy time / makespan 计算，并分别报告 GPU compute、GPU memory、PIM compute、PIM memory。
5. 面积以 synthesis/CACTI 输出为依据；复核论文 T3 buffer 12% overhead 的定义和分母。

论文对应：Sec. 5.1、Table 1、Sec. 5.5、5.6。

测试：空 trace 能耗为零；各项之和等于总能耗；时间、能量、功率单位检查；重叠执行不重复累计 wall time。

### Phase 12：复现 Evaluation

每个实验配置独立保存，不通过修改代码切换方案。

| 论文输出 | 实验矩阵 | 主要指标 |
|---|---|---|
| Fig. 11 | MATH500，9 个 policy/PRM 组合，width 2-8，BW 100/75/50%，GPU/AttAcc/Duplex/ORCHES | normalized speedup |
| Table 1 | 上述文本组合，按 width/length 聚合 | normalized energy efficiency |
| Table 2 | LiveCodeBench，3 个 PRM，BW 100/75/50% | speedup |
| Table 3 | MathVista，width 2/4，short/medium/long | speedup |
| Fig. 12 | MATH500，Qwen2.5-3B policy、Qwen2.5-7B PRM、T2 off，width 2/4/6/8 | AttAcc、A/B/C speedup |
| Table 4 | MATH500，3 个 policy、difficulty 1-5，history alignment off/on | predictor accuracy |
| Fig. 13 | 1B/3B policy、8B PRM，前 10 层 small PRM | timeline、saved latency |
| Table 5 | MATH500，3 x 3 model combinations | context memory saving |
| Table 6 | Qwen2.5-3B policy、3 个 PRM，T1 only/T2 only/T1+T2 | speedup |
| Sec. 5.5/5.6 | 与文本主实验相同 | runtime/area overhead、GPU/PIM utilization |

运行规则：

1. 先运行 microbench 和 calibration，再冻结 `calibration_id`。
2. 每个 problem 单独输出结果；聚合时保存算术平均、几何平均、P50/P95 和 bootstrap 95% CI。
3. 原始 latency/energy 和 normalized 值同时保存。
4. plot 脚本只读 processed CSV/Parquet，不在绘图阶段重新计算模拟结果。
5. 失败、OOM、缺失 trace 不得静默丢弃；进入 manifest 和汇总报告。

### Phase 13：验收、偏差分析和交付

验收分三级：

1. 功能一致：调度、prediction、rollback、KV 状态和 compaction 行为符合论文；全部单元/属性测试通过。
2. 单组件一致：GPU/PIM 单算子与 microbenchmark/AttAcc 的误差在预先定义范围内，误差来源可解释。
3. 端到端趋势一致：search width、SoC bandwidth、T1/T2/T3 消融趋势与论文一致；数值偏差逐项归因。

对论文明确且可直接复现的点，目标 normalized metric 误差不超过 10%；对包含未披露参数的点，
先报告默认值和敏感性区间，再判断是否落在 20% 内。超过阈值时不得只给最终图，必须输出：

```text
workload difference
GPU calibration difference
PIM timing/energy difference
scheduler decision difference
prediction accuracy difference
fragmentation/compaction difference
```

最终交付包括：一键构建、一键 smoke test、一键生成每张图表、原始配置与 manifest、
论文到代码映射、假设清单、校准报告和偏差报告。

## 6. 论文到代码的直接映射

| 论文位置 | 论文机制 | 计划实现 |
|---|---|---|
| Sec. 2.2, Fig. 3 | generation/verification TTC tree | `workload/schema.py`, `collector.py`, `replay.py` |
| Sec. 3.1, Fig. 4-5 | variable parallelism、shared/unique KV | `models/operators.py`, `scheduler/roofline.py` |
| Sec. 3.2, Fig. 6 | branch dependency 和相互等待 | `sim/event.py`, `predictor/pipeline.py` |
| Sec. 3.3 | pruning 产生 fragmentation | `memory/allocator.py` |
| Sec. 4.1, Fig. 7 | GPU/controller die/memory die | hardware YAML、`gpu.py`, `pim.py`, Ramulator C++ |
| Sec. 4.2.1, Eq. 1-4 | T1A offline assignment/co-processing | `scheduler/offline.py` |
| Sec. 4.2.2, Eq. 5-7 | T1B online compensation | `scheduler/online.py` |
| Sec. 4.3.1, Fig. 9a-c | predictor/history alignment/rollback | `predictor/history.py`, `speculation.py` |
| Sec. 4.3.2, Fig. 9d | pipelined verification | `predictor/pipeline.py` |
| Sec. 4.4, Fig. 10 | address cache/reorg/KV buffer | `memory/address_cache.py`, `compactor.py`, `kv_buffer.py` |
| Sec. 5.1 | hardware、simulator、models、datasets | `configs/`, `third_party.lock`, calibration docs |
| Sec. 5.2-5.6 | 主结果和消融 | `configs/experiments/`, `eval/runner.py`, `eval/plots.py` |

## 7. 论文未披露、必须显式处理的项目

1. ORCHES 官方 simulator/artifact 当前未在本仓库中；实现只能基于论文和 AttAcc 上游重建。
2. AGX Orin 的具体 power mode、有效 FP16 throughput、kernel efficiency 和 launch overhead 未给出。
3. PIM 的内部频率、GEMV pipeline latency、controller accum/softmax 吞吐没有完整表格。
4. address cache entry 位宽、容量、相联度和 replacement policy 未给出。
5. shared-KV buffer 容量/端口/带宽及“12% area”的分母未给出。
6. `beta approaches 1` 没有精确 threshold；“每 3-5 steps”也不是唯一触发周期。
7. candidate predictor 的完整打分聚合函数、small PRM 通用配置和 rollback 固定成本未给出。
8. pipelined verification 的最小 prefill token threshold 未给出。
9. MathVista short/medium/long 的边界和部分图的精确聚合方法未给出。
10. AttAcc/Duplex 如何适配 TTC control flow 的全部细节未给出。

处理顺序固定为：先联系作者/查补充材料；仍缺失则继承上游；无法继承则单算子校准；
仍无法获得则使用最简单假设并做敏感性分析。所有选择写入 `docs/assumptions.md`。

## 8. 推荐执行顺序和里程碑

| 里程碑 | 包含 Phase | 可交付结果 |
|---|---|---|
| M1：可构建骨架 | 0-1 | 锁定环境、硬件合同、配置校验 |
| M2：可信基础模型 | 2-4 | TTC trace、GPU baseline、PIM microbench |
| M3：核心 ORCHES | 5-9 | T1/T2/T3 功能与时间线 |
| M4：公平比较 | 10-11 | baselines、能耗、面积、利用率 |
| M5：论文复现 | 12-13 | Fig. 11-13、Table 1-6、偏差报告 |

开始全量 MATH500/LiveCodeBench/MathVista 之前，必须先通过以下门槛：

```text
配置推导正确
地址映射可逆
Ramulator microbench 通过
Eq. (1)-(7) 与参考计算一致
rollback 保持结果一致
compaction 保持数据一致
同一 trace 重放确定
baseline 公平性检查通过
```

## 9. 实施过程如何保持“步骤与论文对应”

后续每完成一个 Phase，同时完成四件事：

1. 提交实现和测试。
2. 更新 `docs/paper-to-code.md`，记录论文页码/公式到函数、配置和测试的映射。
3. 更新 `docs/assumptions.md`，记录新增假设、依据、敏感性范围和是否影响主结论。
4. 生成该 Phase 的最小可复现实验及 manifest，不等到最后才验证。

每个结果记录应能沿以下链路反查：

```text
figure/table cell
  -> processed record
  -> raw per-request simulation
  -> experiment config hash
  -> workload trace hash
  -> scheduler decisions and event timeline
  -> Ramulator counters
  -> source commit + calibration + paper section
```

这条链路是将 ORCHES 作为后续论文 baseline 的最低审计标准。
