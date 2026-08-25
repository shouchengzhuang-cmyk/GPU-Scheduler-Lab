# Phase II: Trace-Driven and Topology-Aware Scheduling

Phase II 在现有 `Scenario -> Simulator -> Scheduler -> Metrics` 边界内增加五类能力：生产 trace normalization、GPU model constraint、层级 topology、reservation/backfill、preemption cost 和 experiment manifest。它没有第二套 engine，也没有外部服务依赖。

## Data flow

```text
Synthetic generator ─┐
Alibaba trace adapter ├─> Scenario ─> Event Queue ─> Scheduler
Mini-AI-Cloud v1 ─────┘                       |          |
                                             v          v
                                      Logical Clock   Placement/Reservation
                                             |          |
                                             +----> Cluster State
                                                       |
                                             Trace + Metrics + Manifest
```

Alibaba-specific columns只存在于 `gpu_scheduler_lab.traces.alibaba`。normalization 完成后，trace workload 与 synthetic workload 走同一个 Scenario、Simulator 和 output 路径。

## Compatibility contract

- 旧 YAML 不写 GPU model 时，GPU 使用 `generic`，Job 没有 model constraint；
- `checkpoint_cost` 与 `restart_cost` 默认 0，保持原抢占时间线；
- Mini-AI-Cloud v1 文件继续可导入，不访问对方 DB、API 或 Python package；
- 旧 `benchmark`、`compare`、`generate`、`import-mini-ai-cloud` 命令继续存在；
- physical/schedulable capacity、fragmentation 和 latency-aware Jain fairness 口径不变。

## Heterogeneous GPU and topology

GPU 包含 `id`、`node_id`、`model` 和 `memory_capacity_gb`。Job 可以声明 exact `gpu_model` 或 `allowed_gpu_models`，两者互斥。没有 model constraint 时只检查 memory。

Node topology 支持 `zone` 与 `rack`。Job topology mode：

- `none`；
- `prefer_same_node`；
- `prefer_same_rack`；
- `require_same_node`；
- `require_same_rack`。

`require_*` 是 feasibility constraint，所有 scheduler 都不能违反。`prefer_*` 只影响 TopologyAware score。TopologyAware 生成有限 domain candidate，不做 GPU 全组合搜索。

## Reservation and backfill

Reservation 包含 head Job、创建时间、预计开始时间、GPU 数和 model constraints，不拥有 GPU。预计开始时间基于当前 running Job 的确定性 completion，是 simulator oracle，不代表生产 scheduler 能精确预知 duration。

后续 Job 必须在 reservation time 前完成才允许 backfill。该规则保护大型 gang 的 guarantee，并允许真正 fit in window 的短任务使用暂时空闲 GPU。它不会保证所有小 Job 的平均等待都下降。

## Preemption cost

采用以下资源语义：

```text
running
  -> checkpointing: GPU occupied, productive runtime frozen
  -> pending: GPU released
  -> restarting: GPU allocated, productive runtime frozen
  -> running: remaining productive runtime resumes
```

checkpoint/restart cost 都进入 turnaround。`wasted_productive_gpu_time` 表示 overhead 阶段占用但不推进 productive runtime 的 GPU-time；模型不包含真实 checkpoint bandwidth 或丢失工作重算。

## Experiment harness

YAML config 定义 workload、schedulers、seeds 和 output directory。每个 run 保存 scheduler、seed、scenario hash、metrics 和完整 result；summary 对核心指标计算 mean、population standard deviation、median 和 P95。

```bash
python -m gpu_scheduler_lab experiment --config experiments/topology.yaml
python -m gpu_scheduler_lab experiment --config experiments/backfill.yaml
python -m gpu_scheduler_lab experiment --config experiments/preemption-cost.yaml
```

输出：

```text
manifest.json
runs.json
summary.csv
summary.json
comparison.png
```

同一个 normalized Scenario 使用 canonical JSON 加 SHA256。Git 不可用时 `git_sha` 为 `null`，实验不失败。

## Deterministic scenarios

- `scenarios/topology.yaml`：Spread 横跨 rack，TopologyAware 选择 same-rack placement；
- `scenarios/backfill.yaml`：一个短 Job 可在 reservation window 内运行，较长 Job 被保守拒绝；
- `scenarios/preemption-zero-cost.yaml`：MVP zero-cost baseline；
- `scenarios/preemption-cost.yaml`：高 checkpoint/restart cost 让 preemptive 的平均等待可能劣于 FIFO。

结论由 simulator 当次输出产生，不硬编码到场景或图表。

## Complexity

- 基础 scheduler 仍以 Node/GPU 遍历为主；
- TopologyAware 候选数随 Node/domain 增长，不使用 $\binom{G}{k}$；
- Backfill 只在 blocked head 时复制 cluster ownership，并按 running completion 扫描；
- Preemptive 只在失败 placement 后构造 projected release state。

10k BinPack/Spread 是 regression benchmark，不代表生产 scheduler throughput。新增策略允许更慢，但 correctness 优先。

## Evidence boundary

Trace-driven simulation 提供更真实的 workload 和 cluster evidence，仍不验证：

- real GPU performance、CUDA throughput 或 OOM；
- NCCL、NVLink、network bandwidth 或 congestion；
- real checkpoint throughput 或 restart latency；
- scheduler control-plane latency、并发 claim 或持久化 recovery；
- Kubernetes、vLLM、MIG、MPS 或 GPU time-sharing。

所有结论都必须写成“在该 trace normalization、Scenario 模型和 scheduler 配置下”。
