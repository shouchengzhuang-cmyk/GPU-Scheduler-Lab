# GPU Scheduler Lab

GPU Scheduler Lab 是一个可复现、可测试、可 benchmark 的 GPU 集群离散事件调度模拟器。它在普通 CPU 电脑上比较 FIFO、BinPack、Spread 和 Priority + Preemption 对 GPU 利用率、显存浪费、碎片、等待时间、公平性与 SLA 的影响。

它是调度算法实验室，不连接真实 NVIDIA GPU、CUDA 或 Kubernetes，也不是生产调度器。

## Why

GPU 集群同时面对设备数量、异构显存、gang 作业、优先级和故障域等约束。集群即使有足够的总资源，也可能因为资源散落、显存型号不合适或低优先级作业占用而无法放置新作业。单看吞吐量无法揭示这些取舍，因此本项目以同一 workload 重放多种策略并输出可审计 trace 和指标。

## Architecture

```text
Scenario / Workload
        |
        v
 Event Queue ---> Simulator Clock
        |               |
        v               v
   Scheduler ------> Cluster State
        |
        v
   Placement
        |
        v
 Trace + Metrics
        |
        v
 Benchmark / Charts
```

模型中的 GPU 是独占设备。一个作业占用 GPU 后，其他作业不能共享该设备；`gpu_memory_gb` 用于容量约束和显存浪费计量。所有多 GPU placement 都是原子的，`gang: true` 进一步启用跨节点 gang 指标。

## Quick start (Ubuntu / WSL)

项目以 Python 3.12 为目标。在 Windows 上请从 Ubuntu WSL 运行：

```bash
cd '/mnt/d/Projects/GPU Scheduler Lab'
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'

.venv/bin/python -m gpu_scheduler_lab compare \
  --scenario scenarios/demo.yaml \
  --schedulers fifo,binpack,spread,preemptive
```

结果默认写入 `results/`：terminal summary、JSON、CSV、策略比较图和 GPU timeline 都来自本次 simulation trace。

单策略运行：

```bash
.venv/bin/python -m gpu_scheduler_lab benchmark \
  --scenario scenarios/mixed.yaml \
  --scheduler fifo
```

生成可复现 workload：

```bash
.venv/bin/python -m gpu_scheduler_lab generate \
  --profile fragmentation --jobs 500 --nodes 20 --gpus-per-node 8 \
  --seed 42 --output scenarios/fragmentation.generated.yaml
```

可用 `--duration-distribution fixed|exponential|lognormal`、`--gpu-count-distribution 1:0.6,2:0.3,4:0.1`、`--gpu-memory-distribution 20:0.7,40:0.3` 和 `--priority-weights low,normal,high,critical` 覆盖 profile 默认分布；arrival rate、duration、training ratio、gang/SLA probability 与 seed 也都是显式参数。

内置 profile：

- `mixed`：inference 与 training 混合；
- `fragmentation`：异构显存请求，强调 stranded capacity；
- `burst`：短时间 inference 突发，强调 tail wait 和抢占。

## Scheduling policies

### FIFO / First Fit

Pending jobs 按 `(arrival_time, job_id)` 稳定排序。每个作业按 scenario 中的 Node/GPU 顺序选择第一组完整可用设备，不抢占。它简单、可解释，但容易造成热点和 head-of-line wait。

### BinPack

BinPack 优先已有占用的 Node，再优先能在放置后留下较小空闲块的 Node；Node 内选择满足显存要求后剩余显存最少的 GPU。确定性 score 顺序为：

```text
empty-node penalty
-> occupied GPU count (descending)
-> eligible free GPU count
-> aggregate memory remainder
-> node id / GPU id
```

这样能保留较完整的空 Node 和大显存 GPU，通常减少碎片；代价是热点、共享故障域和较差的负载分散。

### Spread

Spread 按 Node 当前占用 GPU 数升序排列，并轮询从不同 Node 取一张合格 GPU。它能分散热点和故障域，但多 GPU 作业更容易跨节点，也可能把空闲设备打散，降低后续 gang placement 的成功率和 locality。

### Priority + Preemption

Preemptive policy 使用 BinPack placement，pending queue 按有效优先级降序排列。`low/normal/high/critical` 分别为 0–3；等待每满 30 个逻辑时间单位提升一级，最高到 critical，作为 starvation protection。

高优先级作业放置失败时，只考虑基础优先级严格更低、且当前 running priority 也更低的 running jobs；aging 影响 dispatch 顺序，但不会绕过“不得抢占同级或更高基础优先级”的约束。victim 顺序确定为：低优先级优先、释放 GPU 多者优先、累计运行时间短者优先、Job ID 稳定决胜。被抢占作业释放全部设备、保留累计运行时间并回到 pending；resume 只执行剩余 duration。旧 completion event 由 `run_generation` 隔离，不能释放新 execution 的资源。

这是确定性的 lowest-cost-first 贪心 victim set，不声称求解全局最优组合。

### Gang scheduling

请求 $k$ 张 GPU 的作业只有在完整 placement 存在时才一次性获取 $k$ 张，否则获取 0 张。支持跨 Node，trace 和 `cross_node_gang_placement_count` 会暴露跨节点 gang；不模拟 NCCL、NVLink 或通信成本。

## Metrics

Cluster metrics 包括平均/峰值 GPU utilization、GPU memory utilization、Node utilization、idle GPU time，以及下述 count/memory fragmentation。Job metrics 包括 wait、turnaround、completion、preemption 和 SLA；Scheduling metrics 包括 placement attempts、failed attempts 和 cross-node gang placement。

时间平均指标通过逻辑事件间隔积分，不依赖 `time.sleep()` 或真实 wall clock。

### GPU count fragmentation

令 Node $i$ 有 $c_i$ 张 GPU，其中 $f_i$ 张空闲，$p_i=f_i/c_i$：

$$
F_{count}=\frac{\sum_i c_i \cdot 4p_i(1-p_i)}{\sum_i c_i}
$$

- 0：每个 Node 都是全空或全满，资源块完整；
- 1：所有非空 Node 都恰好半空，Node 内部最零散；
- 范围：$[0,1]$。

它适合比较 pack/spread 造成的 Node partiality，但不是针对某个具体 pending job 的可调度性证明，也不建模互联拓扑带宽。

### GPU memory fragmentation

独占 GPU $g$ 的容量为 $C_g$、实际请求为 $A_g$。被占用设备上的剩余显存不能再服务其他作业，因此：

$$
F_{memory}=\frac{\sum_{g\in occupied}(C_g-A_g)}{\sum_{g\in occupied}C_g}
$$

- 0：无占用，或每个已占用 GPU 的请求正好填满容量；
- 1：理论上已占用设备的显存几乎全部被困住；正数请求下实际值趋近但不会等于 1；
- 范围：$[0,1]$。

综合 `gpu_fragmentation_ratio` 是两者算术平均。这个定义与本项目的 GPU 独占模型一致；若未来支持 time-sharing/MIG，必须更换定义。

### Jain's Fairness Index

对每个 `group`（未指定时用 priority）计算服务率 $x_g$：已完成 GPU-time / 请求 GPU-time，然后计算：

$$
J(x)=\frac{(\sum_g x_g)^2}{n\sum_g x_g^2}
$$

1 表示各组服务率相等，越接近 $1/n$ 越不公平。它衡量 workload group 的相对完成服务率，不代表单个 Job 的等待时间公平。

## Reproducibility

- 所有 arrival/completion 都由 `heapq` 驱动的逻辑时钟处理；
- 同一时刻先 completion、后 arrival，再执行 scheduling；
- placement、pending order 和 victim order 都有稳定决胜字段；
- synthetic generator 使用局部 `random.Random(seed)`；
- `compare` 对同一个不可变 scenario 分别重建状态，不会串用上一策略的 allocation。

## Mini-AI-Cloud integration

本项目与 [Mini-AI-Cloud](https://github.com/shouchengzhuang-cmyk/Mini-AI-Cloud) 通过版本化文件契约联合：Worker GPU inventory 和 Task workload export 可转换为本项目 scenario，结果 JSON 可作为离线 policy study 证据。两边没有 Python import、数据库或服务运行时耦合。

```bash
.venv/bin/python -m gpu_scheduler_lab import-mini-ai-cloud \
  --input scenarios/mini_ai_cloud_demo.json \
  --output scenarios/mini_ai_cloud_demo.generated.yaml
```

字段映射和证据边界见 [Mini-AI-Cloud integration contract](docs/mini-ai-cloud-integration.md)。

## Scale benchmark

`scripts/run_scale_benchmark.py` 固定生成 100 Nodes、800 GPUs、10,000 Jobs，并对同一个 seed workload 运行 BinPack 和 Spread：

```bash
.venv/bin/python scripts/run_scale_benchmark.py
```

脚本输出实际 wall time 和模拟指标，不把 in-process simulator throughput 描述成生产 scheduler、数据库或真实 GPU 性能。

2026-08-25 在 Ubuntu 24.04 WSL、Python 3.12.3 上的固定 seed `20260825` 实测如下（100 Nodes / 800 GPUs / 10,000 Jobs；重新运行会因机器负载产生不同 wall time，但逻辑指标应一致）：

| Scheduler | Simulator elapsed | Completion | Avg GPU util | Avg wait | P95 wait | Fragmentation | SLA violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| BinPack | 8.130 s | 1.000 | 0.5520 | 0.035 | 0.000 | 0.2777 | 0.0000 |
| Spread | 7.409 s | 1.000 | 0.5520 | 0.000 | 0.000 | 0.5594 | 0.0000 |

双策略 simulation 段 wall time 为 15.565 s；含 Python 启动、workload 生成和 JSON 写出的完整进程为 15.99 s，峰值 RSS 约 37,580 KiB。原始结果由脚本重新生成，不提交 `results/`。

## Development and CI

```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/mypy .
.venv/bin/pytest
```

GitHub Actions 在 Ubuntu + Python 3.12 上执行同样四项检查，不需要 GPU 或外部服务。

## Limitations

这是离散事件 simulation。它没有声称验证：

- 真实 NVIDIA GPU、CUDA kernel、显存分配器或 OOM；
- NCCL、NVLink、PCIe、网络通信或真实 distributed training；
- MIG、GPU time-sharing、功耗和热管理；
- Kubernetes device plugin、scheduler plugin 或 production scheduler behavior；
- vLLM inference throughput、TTFT、TP/PP 性能；
- Mini-AI-Cloud 的 PostgreSQL transaction、lease、fencing 或 runtime lifecycle。

Wall-clock benchmark 只衡量本机 Python 事件循环和策略实现，不能外推真实集群吞吐。

详细设计见 [docs/design.md](docs/design.md)。
