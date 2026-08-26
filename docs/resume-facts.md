# Resume facts: GPU Scheduler Lab

本文件只记录能由仓库、测试和正式 study bundle 复核的事实。所有数值属于确定性离散事件 simulation，不是生产 scheduler、真实 GPU 或 Kubernetes 性能。

## Scope and model

- Python 3.12 离散事件 simulator 使用统一的 `Scenario -> Simulator -> Scheduler -> Metrics` 路径比较策略。
- 正式 study 的 synthetic fleet 为 8 个节点、每节点 8 张 GPU，共 64 张模型设备；每个 run 生成 240 个作业。
- 模型覆盖异构 GPU/显存、node/rack/zone topology、多租户 queue/fair share、借用与 reclaim、elastic gang、checkpoint/restart 和 dynamic/revocable fleet。
- 正式四类策略固定为 `binpack`、`topology-aware`、`historical-drf` 和 `fairshare-reclaim`。

## Correctness and reproducibility

- 当前源码树由 pytest 收集并通过 `186` 个测试；其中 12 条 Phase III 调度不变量关联语义断言，3 个 canonical scenario 固定 deterministic golden baseline。
- study runner 支持 stable run ID、bounded retry、部分恢复、warm-up、多 seed、one-at-a-time sensitivity 和单机制 ablation。
- 正式 `make reproduce-study` 在 clean commit `c4e97beb36a1cf45b7adc676be7a80934807d315` 完成 180 runs，使用 3 个 seeds、5 个 sensitivity variables 和 4 个 ablations。
- 该 bundle 生成 3 张 CSV 表、3 张 PNG 图、Markdown report，并由 `hashes.sha256` 验证 373 个产物；manifest 记录 `dirty_tree: false`。
- Mini AI Cloud v1 输入具有 JSON Schema、golden/breaking compatibility fixtures、未知字段审计、CPU-only/health filter、时间/优先级/model/topology 映射；结果 handoff 固定标记 `SIMULATED`。

## Formal baseline snapshot

下表来自正式 bundle 的 `summary.json`，每项为 3 个 seed 的 mean 与 population stddev：

| Policy | Completion | Avg GPU utilization | P95 wait | Jain service-quality fairness |
|---|---:|---:|---:|---:|
| binpack | 1.0000 ± 0 | 0.766684 ± 0.058342 | 406.812 ± 76.198 | 1.000000 ± 0 |
| topology-aware | 1.0000 ± 0 | 0.766684 ± 0.058342 | 406.812 ± 76.198 | 1.000000 ± 0 |
| historical-drf | 1.0000 ± 0 | 0.784393 ± 0.039683 | 443.754 ± 77.286 | 0.979657 ± 0.006574 |
| fairshare-reclaim | 1.0000 ± 0 | 0.760870 ± 0.041686 | 436.073 ± 67.585 | 0.976258 ± 0.005328 |

这些数值只描述冻结的 synthetic workload family。它们不证明某策略在真实集群中更快，也不构成统计显著性结论。

## Limitations

- 未运行 NVIDIA GPU、CUDA、NCCL、NVLink、真实 checkpoint bandwidth、Kubernetes scheduler plugin 或生产 control plane。
- logical time 不是 wall-clock latency；simulator elapsed time 不是生产 scheduler throughput。
- GPU 型号、tenant、topology cost、revocation 和 workload 参数均为显式 simulation 输入，不是生产采样。
- `0.3.0` 仅完成 release 准备；未自动创建 tag、GitHub Release 或部署。
