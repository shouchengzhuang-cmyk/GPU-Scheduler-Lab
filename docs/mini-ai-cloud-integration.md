# Mini-AI-Cloud integration contract

## Boundary

联合采用 `mini-ai-cloud.gpu-scheduler-lab/v1` JSON 文件契约。规范文件为 [`contracts/mini-ai-cloud-v1.schema.json`](../contracts/mini-ai-cloud-v1.schema.json)，运行时 adapter 以无第三方依赖的等价检查验证所有被消费字段。GPU Scheduler Lab 不连接 Mini-AI-Cloud 数据库/API，不导入其内部包，也不修改其调度状态。

Mini-AI-Cloud 当前概念到 simulator 的映射：

| Mini-AI-Cloud | GPU Scheduler Lab | 说明 |
|---|---|---|
| Worker `id` | Node `id` | Worker 作为资源/故障域 |
| Worker schedulable state | Node `schedulable` | false 时保留 physical inventory，但不计入可调度指标分母 |
| Worker labels | Node topology | `zone`、`rack` 等 string label 原样映射 |
| GPU `device_uuid` | GPU `id` | 保持具体设备 identity |
| `memory_total_mb` | `memory_capacity_gb` | 除以 1024 |
| GPU `model` | GPU `model` | 缺省时使用 `generic`；用于型号可行性检查 |
| unhealthy GPU | filtered | 不进入 inventory，并记录过滤数量 |
| Task `queued_at` / `arrival_time` | Job `arrival_time` | 归一化到最早 GPU Task 为 0 |
| `duration_seconds` / `timeout_seconds` | Job `duration` | 前者优先；这是实验输入，不是运行时预测 |
| `gpu_count` | Job `gpu_count` | CPU-only Task 会被过滤并记录数量 |
| `gpu_memory_mb` | Job `gpu_memory_gb` | GPU Task 必须为正数 |
| priority 0–100 | low/normal/high/critical | 0–24 / 25–74 / 75–89 / 90–100 |
| `project_id` | Job `group` | 用于按完成率与延迟效率计算 Jain fairness |
| workload type | inference/training | `training`/`batch_job` 映射 training，其余映射 inference |
| label `gpu_scheduler_lab/gang=true` | `gang=true` | 多 GPU Task 也默认 gang |
| `gpu_model` / `allowed_gpu_models` | 同名 Job 约束 | 两者互斥；用于设备型号过滤 |
| label `gpu_scheduler_lab/topology` | `topology_mode` | 接受 `none`、`prefer_same_node`、`prefer_same_rack`、`require_same_node`、`require_same_rack` |

## Compatibility rules

- `contract_version` 必须精确为 `mini-ai-cloud.gpu-scheduler-lab/v1`；未知版本明确失败，不猜测迁移。
- v1 已知且被 adapter 消费的字段执行类型、范围、唯一性与互斥检查。
- 未知字段允许出现，以便生产者在 v1 内向前扩展；adapter 不解释其语义、不影响调度，只把未知字段名记录到 scenario metadata 或 Job `source_metadata`，未知值不会复制进结果。
- CPU-only Task 在时间基线计算前过滤；因此它不会把 GPU workload 的逻辑起点提前。
- 只有 health 精确为 `healthy`（缺省也是 `healthy`）的设备进入 inventory。
- 数值时间直接视为 epoch/逻辑秒；带时区 ISO-8601 转为 UTC epoch；无时区 ISO-8601 明确按 UTC 解释。
- priority 固定映射为 0–24 low、25–74 normal、75–89 high、90–100 critical。

`tests/fixtures/mini_ai_cloud/v1-golden.json` 与对应 expected snapshot 冻结完整映射；`v1-breaking.json` 证明破坏 GPU memory 必填约束时会明确失败。未来 v2 必须使用新 `contract_version` 和独立 schema；adapter 若增加 v2，仍必须保留 v1 golden compatibility test。

## Import and reproduce

```bash
python -m gpu_scheduler_lab import-mini-ai-cloud \
  --input tests/fixtures/mini_ai_cloud/v1-golden.json \
  --output build/mini-ai-cloud.golden.yaml
python -m gpu_scheduler_lab compare \
  --scenario build/mini-ai-cloud.golden.yaml \
  --schedulers binpack,topology \
  --output-dir build/mini-ai-cloud-result
```

## Result handoff

`benchmark` / `compare` JSON 遵循 [`contracts/result-handoff-v1.schema.json`](../contracts/result-handoff-v1.schema.json)，固定包含：

- `contract_version: gpu-scheduler-lab.result/v1`；
- `evidence_kind: SIMULATED`；
- 至少一条 simulation limitation；
- scheduler、完整 metrics、逐 Job outcome 和 trace。

Mini-AI-Cloud 可把它作为离线 policy study artifact 保存，但不得把 simulation completion、SLA 或 utilization 当成真实 Task/Worker 运行结果，也不得据此自动切换线上策略。

## Non-goals

- 不从生产数据库直接导出或回写；
- 不共享 SQLAlchemy/Pydantic 内部类；
- 不替代 Mini-AI-Cloud 的 transaction、reservation、lease、execution fencing；
- 不声称模拟结果验证真实 Docker/Kubernetes/GPU runtime。
