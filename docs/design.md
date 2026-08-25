# Design notes

## 1. Why discrete-event simulation

调度研究关注状态发生变化的时点：Job arrival、completion、preemption 和 resume。事件之间没有需要逐 tick 更新的行为，因此 priority queue 将复杂度集中在真实变化处，也让 10,000 Job benchmark 不必 `sleep` 或扫描空闲时间片。wall clock 只用于报告 simulator 自身运行耗时，不参与结果。

## 2. Job lifecycle

```text
not arrived -> pending -> running -> completed
                    ^        |
                    |        v
                    +-- preempted
```

`accumulated_runtime` 只在 preempt/complete 更新。每次 start/resume 增加 `run_generation`，completion event 携带 generation；被抢占前排入的旧 completion event 因 generation 不匹配而被忽略。

## 3. Scheduler interface

`Scheduler.place(cluster, job)` 是无副作用函数：返回恰好 `gpu_count` 个 GPU ID，或 `None`。Engine 是唯一修改 Cluster/Job lifecycle 的组件。这一边界让 policy 容易单测，并保证 gang all-or-nothing。

`pending_key(job, now)` 控制队列顺序。普通策略使用 arrival + stable Job ID；preemptive policy 使用 aged effective priority。

每个事件时刻对稳定排序后的 pending snapshot 单次遍历，避免每成功放置一个 Job 就从队头重新扫描；本轮新增的 preemption victims 只获得一次即时 resume 尝试。这样保留 deterministic backfill，同时避免 burst workload 下明显的重复全队列工作。

## 4. Preemption semantics

只有 placement 失败且存在优先级更低的 running jobs 时才抢占。victim 采用确定性 cost order 的最短可行前缀。被抢占作业保留已运行时间、释放所有 GPU、回 pending；不模拟 checkpoint I/O 或 restart overhead，这是已知乐观假设。

Aging 每等待 30 个逻辑时间单位提升一级，防止 low/normal 在资源释放点永久被新 high jobs 越过。它不会绕过 victim 必须具有更低基础优先级的约束；dispatch 时的 aged priority 会固定到该 running attempt，避免同一时刻反向抢占。

## 5. Gang placement

Engine 对所有 placement 都要求原子性；gang flag 表示 workload 明确要求协同启动，并启用跨 Node gang 统计。MVP 允许跨 Node，但不改变 duration，也不模拟 collective communication penalty。

## 6. Fragmentation

Count fragmentation 使用 Node partiality $4p(1-p)$ 的容量加权平均；Memory fragmentation 使用独占 GPU 上 stranded memory 的比例；综合指标取二者平均。详细公式、0/1 语义和局限见 README。

## 7. Determinism

- Event key：time、complete-before-arrival order、monotonic sequence；
- FIFO tie：arrival、Job ID；
- Placement tie：Node ID、GPU ID；
- Victim tie：priority、freed GPU count、accumulated service、Job ID；
- Workload randomness：局部 seeded RNG。

`elapsed_seconds` 不参与 determinism assertion；trace、Job outcome 和 metrics 必须相等。

## 8. Difference from a real GPU scheduler

真实 scheduler 还要处理并发 claim、持久化 reservation、Worker lease、device health drift、runtime startup failure、checkpoint 成本、网络拓扑和资源遥测陈旧。本 simulator 使用单线程内存状态，目标是隔离 policy trade-off，不验证控制面一致性或 GPU runtime。

Mini-AI-Cloud 负责 production-minded control-plane experiments；GPU Scheduler Lab 通过稳定文件契约接收其 inventory/workload 快照用于离线策略研究，两者证据不可互相替代。
