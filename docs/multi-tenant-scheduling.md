# Multi-Tenant Scheduling Semantics

## Guarantee and limit

Guarantee 表示 queue 在有 demand 时应当恢复到的 entitlement，不是静态 reservation。Limit 是硬上限，direct usage 与全部 descendant usage 都不能越过对应 ancestor limit。Guarantee 为 0 的 queue 只能借用，limit 等于 guarantee 的 queue 不能借用。

Queue 的 aggregate usage 来自后代 Job allocation。父 queue 的 average usage、peak、borrowed time、completion 和 rejection 也按后代聚合，避免层级指标与 leaf 结果对不上。

## Borrowing

Job 启动时，allocation 先消耗 queue 尚未使用的 guarantee，剩余部分记为 borrowed units。Borrowing 不修改 guarantee，也不会在下一轮变成永久 entitlement。`borrowing_enabled: false` 时，projected aggregate usage 超过 guarantee 就停止 allocation。

## Reclaim

Reclaim 只服务于低于 guarantee 的 runnable queue。Victim queue 必须不同、允许 reclaim，并且 victim allocation 含 borrowed units。系统不会为了让 incoming queue 自己借更多资源而抢走 sibling 的 guarantee 内 workload。

Elastic victim 会先缩到 min。固定 Job 或仍需释放更多 GPU 时，reclaim 进入已有 checkpoint state machine。完整 projected placement 在 checkpoint 期间保留，target 启动后才释放 suspended victims。

## DRF score

GPU units 和 requested GPU memory 是两个独立 accounting dimension：

```text
gpu_share = queue_gpu_units / schedulable_gpu_units
memory_share = queue_requested_memory / schedulable_gpu_memory
weighted_dominant_share = max(gpu_share, memory_share) / queue_weight
```

DRF score 只决定 dispatch opportunity。型号、显存、拓扑、gang 和当前 free state 仍由 placement scheduler 检查。

## Historical debt

History 用 logical time 积分 queue 收到的 GPU service，并用配置的 half-life 衰减旧值。Sibling 内 `historical_service / weight` 最小的 queue debt 为 0，其余 queue 的正 debt 表示历史上服务更多。Historical DRF 在 instantaneous usage 相同或接近时让 under-served sibling 先运行。

## Starvation

Starvation 的定义是 admitted runnable Job 在首次启动前等待超过 `fairshare.starvation_threshold` 个逻辑时间单位。该指标不推断原因，只报告超过阈值的 Job 数。

## Accounting limits

`gpu_memory_gb` 按 Job 请求量记账，不按 GPU capacity 记账。Model weight 默认 1；用户配置其他值时，它仍只是资源分配政策输入。Simulator 不从型号名称推断算力、价格或训练速度。
