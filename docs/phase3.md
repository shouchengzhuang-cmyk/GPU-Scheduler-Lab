# Phase III: Multi-Tenant GPU Fleet Scheduling

Phase III 把调度决策拆成五层：Admission 判断 workload 能否进入系统；Queue/FairShare 决定资源机会给谁；Allocation 处理 borrowing 和 reclaim；Placement 选择具体 GPU；Fleet event 改变可用容量。旧场景没有 queue、elastic 或 fleet 字段时，仍按 Phase I/II 行为运行。

## Scenario additions

Queue 定义来自 scenario：

```yaml
queues:
  - id: research
    parent: root
    weight: 1
    guaranteed: {gpu_units: 2}
    limit: {gpu_units: 4}
    borrowing_enabled: true
    reclaimable: true

admission: {mode: quota-aware}
fairshare: {half_life: 300, starvation_threshold: 300}
```

没有 `queue` 的 Job 进入 `root/default`。Loader 会拒绝重复 ID、cycle、missing parent、负数或非有限 quota、guarantee 大于 limit，以及 child guarantee 总量超过 parent limit 的配置。

## Admission

`permissive` 接受结构合法且物理上可能满足的请求。`quota-aware` 还会拒绝最小请求超过 queue 或 ancestor hard limit 的 Job。两种模式都不会因为 GPU 正忙而拒绝 Job。

结果分别记录 submission、admission、queue wait 和 rejection reason。Admission 目前是同步的，所以正常 Job 的 admission wait 为 0；这个字段单独保留，避免以后引入 admission delay 时修改 scheduling wait 的含义。

## Allocation and reclaim

Queue guarantee 不提前占 GPU。Sibling 空闲时，开启 borrowing 的 queue 可在 ancestor 和自身 limit 内使用空闲 entitlement。每个时间区间记录 guaranteed usage、borrowed usage 和 unused entitlement。

当有 runnable queue 低于 guarantee，reclaim policy 只考虑其他 queue 中标记为 borrowed 的 allocation。Victim 顺序包含 borrowed amount、priority、可释放的适配 GPU、collateral、remaining work、checkpoint/restart cost 和稳定 ID。请求自身超过 guarantee 时，不能抢走 sibling 的 in-guarantee workload。

## Fair share

Instantaneous DRF 比较 GPU units 与请求显存的 dominant share，再除以 queue weight。Historical DRF 增加 logical-time service debt。Guarantee deficit 优先，之后依次比较 debt、weighted dominant share、priority、arrival 和 Job ID。

型号权重只改变记账值。它不是 throughput、价格或真实硬件价值模型，实验结论必须把权重写成 policy assumption。

## Elastic gang

```yaml
elastic:
  min_replicas: 4
  preferred_replicas: 8
  max_replicas: 16
  scaling_efficiency:
    4: 0.9
    8: 1.0
    16: 0.82
```

Job 至少取得 min 才能原子启动。默认 work rate 等于 replica 数；显式 efficiency curve 会把速率改为 `replicas * efficiency`。Scale-up 只发生在事件边界，目标先到 preferred，单个时间点最多执行一次扩容，避免连续振荡。Reclaim 先从 borrowed elastic allocation 缩到 min，低于 min 只能整体 suspend 或 preempt。

## Dynamic fleet

`NODE_JOIN` 增加此前 unavailable 的容量；`NODE_DRAIN` 禁止新 placement，但不终止已有 Job；`NODE_FAIL` 立即使 Node unavailable；`NODE_RECOVER` 恢复 Node；`CAPACITY_REVOKE` 和 `CAPACITY_RETURN` 对 revocable capacity 执行同类撤回与返回语义。

Forced loss 会保留已经结算的 productive work，Job 重新进入 pending/restart 路径。这个 optimistic recovery model 没有模拟 durable checkpoint interval。Revocable GPU 在撤回前仍是普通独占 GPU，没有第二套 ownership。

## Required scenarios

- `multi-tenant-borrow-reclaim.yaml` 比较 borrowing 与 reclaim。
- `historical-fairshare.yaml` 比较 instantaneous 和 historical DRF。
- `fixed-gang-changing-capacity.yaml` 与 `elastic-gang.yaml` 比较固定和弹性 gang。
- `stable-fleet.yaml` 与 `revocable-fleet.yaml` 比较稳定和撤回容量。

Experiment 额外输出 queue share、borrowed capacity、fair-share debt、elastic replicas 和 fleet capacity 五张 timeline。Manifest 保存 queue config hash、allocation policy、fairshare config、fleet event hash 和 elastic model version。

## Evidence boundary

Phase III 只验证显式模型下的 policy semantics。它没有验证真实 GPU scaling、NCCL、network congestion、CUDA checkpoint throughput、Kubernetes queue、Spot interruption probability、真实 Alibaba tenant hierarchy 或硬件价值比例。Synthetic tenant overlay 会写入 `tenant_assignment: synthetic_overlay`，不能当作 Alibaba 的真实租户字段。
