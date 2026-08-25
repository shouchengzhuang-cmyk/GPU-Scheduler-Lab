# Design notes

## 1. Why discrete-event simulation

调度研究关注状态发生变化的时点：Job arrival、completion、checkpoint、restart、preemption 和 resume。事件之间没有需要逐 tick 更新的行为，因此 priority queue 将复杂度集中在真实变化处，也让 10,000 Job benchmark 不必 `sleep` 或扫描空闲时间片。wall clock 只用于报告 simulator 自身运行耗时，不参与结果。

`SCHEDULER_TICK` 是 aging bookkeeping event，不是 workload activity。当集群没有 running、checkpointing 或 restarting Job，队列只剩 aging tick 时，pending Job 已在全空闲 schedulable capacity 上放置失败，tick 无法改变可行性，因此 simulator 丢弃这些 tick，不让 policy 内部定时器污染 material horizon、utilization denominator 或 idle GPU time。

## 2. Job lifecycle

```text
not arrived -> pending -> running -> completed
                    ^        |
                    |        v
                    + checkpointing
                    |        |
                    |        v
                    +-- restarting
```

`accumulated_runtime` 只在 preempt/complete 更新。每次 start/resume 增加 `run_generation`，completion event 携带 generation；被抢占前排入的旧 completion event 因 generation 不匹配而被忽略。

## 3. Scheduler interface

`Scheduler.place(cluster, job)` 是无副作用函数：返回恰好 `gpu_count` 个 GPU ID，或 `None`。Engine 是唯一修改 Cluster/Job lifecycle 的组件。这一边界让 policy 容易单测，并保证 gang all-or-nothing。`prepare()` 只给 reservation scheduler 一个调度轮次的只读视图，`on_job_started()` 只在 Engine 已提交 allocation 后更新 policy bookkeeping。

`pending_key(job, now)` 控制队列顺序。普通策略使用 arrival + stable Job ID；preemptive policy 使用 aged effective priority。

每个事件时刻对稳定排序后的 pending snapshot 单次遍历，避免每成功放置一个 Job 就从队头重新扫描；本轮新增的 preemption victims 只获得一次即时 resume 尝试。这样保留 deterministic backfill，同时避免 burst workload 下明显的重复全队列工作。

## 4. Preemption semantics

只有 placement 失败且存在优先级更低的 running jobs 时才抢占。victim 采用确定性 greedy order，考虑适配 GPU、collateral GPU、剩余 productive runtime、checkpoint/restart cost 和稳定 ID。

抢占开始时 productive runtime 冻结。checkpoint delay 内 GPU 仍归 victim，checkpoint-complete 才释放。resume 先分配 GPU，再经历不推进 productive runtime 的 restart delay。cost 为 0 时保持 MVP 的即时释放和恢复语义。`run_generation` 同时隔离 checkpoint 前的旧 completion event。

一次抢占若需要 victim checkpoint，Engine 会为 incoming Job 保留完整 projected placement，包括 victim 将释放的设备和 placement 中原本空闲的设备；reservation 不占 owner、不计 busy time。已先完成 checkpoint 的 victim 暂时挂起，等 incoming Job 真正取得完整 gang placement 后再进入 pending，避免 victim 或同级 pending Job 抢走部分资源。

同一逻辑时间按 `JOB_COMPLETE`、`JOB_CHECKPOINT_COMPLETE`、`JOB_RESTART_COMPLETE`、`JOB_ARRIVAL`、`SCHEDULER_TICK` 排序，再执行 scheduling。

Aging 每等待 30 个逻辑时间单位提升一级，防止 low/normal 在资源释放点永久被新 high jobs 越过。它不会绕过 victim 必须具有更低基础优先级的约束；dispatch 时的 aged priority 会固定到该 running attempt，避免同一时刻反向抢占。

## 5. Gang placement

Engine 对所有 placement 都要求原子性；gang flag 表示 workload 明确要求协同启动，并启用跨 Node gang 统计。MVP 允许跨 Node，但不改变 duration，也不模拟 collective communication penalty。

## 6. Fragmentation

Count fragmentation 使用 Node partiality $4p(1-p)$ 的容量加权平均；Memory fragmentation 使用独占 GPU 上 stranded memory 的比例；综合指标取二者平均。详细公式、0/1 语义和局限见 README。

Cluster 把四种容量分开：physical capacity 是全部 Node/GPU inventory；potential capacity 是当前可调度 Node 与尚未发生的 join/recover/return，只用于 admission 和 synthetic tenant overlay；schedulable capacity 是当前 available、schedulable 且含 GPU 的 Node，用于 placement、fair-share 与 fragmentation；active capacity 是全部 schedulable GPU 加上 draining Node 中仍被占用的 GPU，用于 utilization、idle time、memory、Node、stable/revocable 和 fleet timeline 的区间积分。`schedulable: false`、unavailable、零 GPU Worker 和已经排空的 draining Node 都不会稀释对应实验指标。

## 7. Determinism

- Event key：time、complete-before-arrival order、monotonic sequence；
- FIFO tie：arrival、Job ID；
- Placement tie：Node ID、GPU ID；
- Victim tie：priority、suitable GPU count、total allocated GPU count、accumulated service、Job ID；
- Workload randomness：局部 seeded RNG。

`elapsed_seconds` 不参与 determinism assertion；trace、Job outcome 和 metrics 必须相等。

## 8. Difference from a real GPU scheduler

真实 scheduler 还要处理并发 claim、持久化 reservation、Worker lease、device health drift、runtime startup failure、checkpoint 成本、网络拓扑和资源遥测陈旧。本 simulator 使用单线程内存状态，目标是隔离 policy trade-off，不验证控制面一致性或 GPU runtime。

Mini-AI-Cloud 负责 production-minded control-plane experiments；GPU Scheduler Lab 通过稳定文件契约接收其 inventory/workload 快照用于离线策略研究，两者证据不可互相替代。

## 9. Phase II GPU model and topology

`GPU.is_compatible(job)` 集中检查 memory 与 exact/allowed model，`GPU.can_host(job)` 再叠加 free 状态。`require_same_node` 与 `require_same_rack` 是硬约束；普通 scheduler 遇到 required topology 时复用 topology placement。

`topology_distance()` 集中定义 same-node 0、same-rack 1、same-zone 2、cross-zone 3。缺失 rack 或 zone 时，不同 Node 使用隔离的 unknown domain，不会被错误归为同一机架。

TopologyAware 构造 global、node、rack、zone 和每个 seed Node 的邻近候选，不枚举 $\binom{G}{k}$ 全组合。score 顺序为 required feasibility、preferred domain count、maximum/average pairwise distance、count fragmentation delta、memory waste、stable GPU IDs。它是确定性启发式，不声称全局最优。

## 10. Phase II reservation and backfill

Backfill scheduler 为 FIFO head 保存不占用 GPU 的 `Reservation`。它复制当前 ownership，按 running completion 顺序释放资源，并用 topology placement 计算最早可行时间。后续 Job 只有满足 `now + remaining_duration <= estimated_start_time` 才能启动。

这是 conservative EASY-style 规则。它保证 reservation 不被 backfill 延迟，但可能推迟不能在 window 内完成的小 Job。estimate 直接读取 simulation duration，是 oracle-style limitation。

## 11. Phase II metrics and reproducibility

Topology metrics 分类 same-node、same-rack、cross-rack、cross-zone，并对 GPU pair 的层级距离求平均。Reservation 与 preemption overhead 都输出独立 audit metrics；require violation 与 reservation delay violation 正常必须为 0。

Experiment harness 使用 canonical scenario JSON 的 SHA256 证明各 scheduler 输入一致，并保存 Git SHA、Python version、trace source/version、seed、metrics 和聚合结果。manifest timestamp、Git SHA 与 wall-clock elapsed 不参与 deterministic result assertion。

复杂度热路径：TopologyAware 只构造有限 domain 候选；Backfill 只在 blocked head 时复制一次 cluster；Preemptive 只在失败 placement 时构造 projected cluster。10k BinPack/Spread benchmark 继续作为稳定 regression baseline。

## 12. Phase III separates admission, allocation, and placement

Admission 在 Job arrival 时校验 queue、硬上限和物理上不可能满足的 GPU 请求。暂时忙碌不构成拒绝理由。Allocation policy 决定哪个 queue 和 Job 获得下一次机会，placement scheduler 决定 GPU ID。FairShareScheduler 通过组合 TopologyAware 实现这条边界，没有复制 placement 逻辑。

Queue usage 同时向全部 ancestor 聚合。每个 allocation 记账 `gpu_units` 和请求显存，型号权重只作为 policy 输入，不代表硬件性能。Guarantee 是 entitlement，limit 是硬约束；超过 guarantee 的部分记为 borrowed usage。

## 13. DRF and logical-time history

Weighted DRF score 集中定义为：

$$
\frac{\max(U_{gpu}/C_{gpu}, U_{mem}/C_{mem})}{weight_q}
$$

历史 service 只使用 simulation logical time。设 $\lambda=\ln(2)/h$，在长度为 $\Delta t$ 的区间内 queue 以常数速率 $r_q$ 获得 GPU service，则旧 service 与区间内新增 service 使用同一连续衰减模型：

$$
H_q(t+\Delta t)=H_q(t)e^{-\lambda\Delta t}+r_q\frac{1-e^{-\lambda\Delta t}}{\lambda}
$$

因此，把没有速率变化的区间拆成多个 event interval 不会改变历史 service。

同一 parent 下，`fairshare_debt` 等于 queue 的 `historical_service / weight` 减去 sibling 中的最小值。正数表示历史上收到过更多服务，调度顺序会暂时后移；负 debt 不在当前基线中出现，最欠服务的 sibling 为 0。

## 14. Reclaim reuses preemption fencing

Reclaim 和 priority preemption 使用同一个 projected-placement state machine，trace 的 structured reason 区分 `PREEMPT_PRIORITY`、`PREEMPT_RECLAIM` 与 `PREEMPT_CAPACITY_REVOKE`。Reclaim victim 必须来自层级边界另一侧且该 branch 高于自身 entitlement floor。Planner 在一个 projected transaction 中先尝试缩减 borrowed elastic replica，再按需 checkpoint 整个 victim；只有完整 incoming placement 已存在时才提交，不能留下无效的部分 shrink。

多 victim reclaim 先计算完整 incoming placement 并保留全部 GPU。先完成 checkpoint 的 victim 保持 suspended，incoming 启动后才回到 pending，其他 Job 看不到这部分 reservation。

## 15. Elastic work and fleet events

Fixed Job 继续使用 duration。Elastic Job 的 total work 是 `duration * preferred_replicas`，实际速率是 `replicas * configured_efficiency`。Resize 先结算当前 work，再原子更新 GPU ownership、增加 generation 并重排 completion，旧 event 无法提前完成 Job。

同一时间戳的顺序固定为 completion、checkpoint completion、restart completion、capacity addition/recovery、capacity drain/failure/revoke、Job arrival、scheduler tick。Drain 后现有 Job 继续运行；fail 和 revoke 立即使 allocation 失效。Recovery 模型保留已完成 work，并加 configured restart cost。真实故障可能丢失最近 durable checkpoint 之后的工作，simulator 没有验证这部分。

Dynamic fleet 的 utilization denominator 按事件区间积分 active capacity，不使用 simulation 结束时的单一容量回算整个 horizon。

Queue 的 `guaranteed_share_satisfaction` 同样按区间积分，但 denominator 只计 `min(guarantee, aggregate runnable demand)`。无 demand 的时间不惩罚 queue；没有 entitlement demand 时结果定义为 1。
