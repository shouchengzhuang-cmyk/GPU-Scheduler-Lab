# Mini-AI-Cloud integration contract

## Boundary

联合支持两版 JSON 文件契约：兼容版 `mini-ai-cloud.gpu-scheduler-lab/v1` 和类型化版 `mini-ai-cloud.gpu-scheduler-lab/v2`。规范文件分别为 [`contracts/mini-ai-cloud-v1.schema.json`](../contracts/mini-ai-cloud-v1.schema.json) 与 [`contracts/mini-ai-cloud-v2.schema.json`](../contracts/mini-ai-cloud-v2.schema.json)，运行时 adapter 以无第三方依赖的等价检查验证所有被消费字段。GPU Scheduler Lab 不连接 Mini-AI-Cloud 数据库/API，不导入其内部包，也不修改其调度状态。

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

v2 在 v1 文件形状上增加类型化 accelerator 字段。为保持传输兼容，设备数组和 count/memory 字段在本版本仍沿用 `gpu_devices`、`gpu_count`、`gpu_memory_mb` 名称；`vendor`、`kind` 和新的任务约束才是厂商类型事实，adapter 不根据旧字段或型号猜测厂商。

| v2 字段 | 内部字段 | 规则 |
|---|---|---|
| Device `vendor` / `kind` | GPU `vendor` / `kind` | 只接受 `nvidia+gpu`、`huawei-ascend+npu` |
| Device `runtime_profiles` | GPU `runtime_profiles` | 非空 string 的去重数组 |
| Device `capabilities` | GPU `capabilities` | 非空 string 的去重数组 |
| Task `allowed_vendors` / `allowed_kinds` | Job 同名约束 | 只保存明确值，不从 model 推断 |
| Task `allowed_models` | Job `allowed_models` | 与 legacy `allowed_gpu_models` 分开 |
| Task `required_capabilities` | Job 同名约束 | typed placement 必须满足全部 capability |
| Task `runtime_profile` | Job `runtime_profile` | 可为 null 或非空 string |
| Task `selection_policy` | Job `selection_policy` | v2 固定为 `any` |

## Compatibility rules

- `contract_version` 必须精确为 v1 或 v2；未知版本明确失败，不猜测迁移。
- v1 已知且被 adapter 消费的字段执行类型、范围、唯一性与互斥检查。
- v1 设备明确落为 `vendor=unknown`、`kind=gpu` 并设置 `accelerator_metadata_inferred=true`；即使 model 名包含 Ascend，也不得自动推断为华为昇腾。
- v2 设备必须显式提供 vendor、kind、model、runtime profiles 和 capabilities；vendor-kind 错配 fail closed。
- 未知字段允许出现，以便生产者在同一版本内向前扩展；adapter 不解释其语义、不影响调度，只把未知字段名记录到 scenario metadata 或 Job `source_metadata`，未知值不会复制进结果。
- CPU-only Task 在时间基线计算前过滤；因此它不会把 GPU workload 的逻辑起点提前。
- 只有 health 精确为 `healthy`（缺省也是 `healthy`）的设备进入 inventory。
- 数值时间直接视为 epoch/逻辑秒；ISO-8601 时间必须采用 `YYYY-MM-DDTHH:MM:SS`（可带小数秒）并可选 `Z` 或 `±HH:MM` 时区；带时区值转为 UTC epoch，无时区值明确按 UTC 解释。
- priority 固定映射为 0–24 low、25–74 normal、75–89 high、90–100 critical。

v1/v2 各自拥有 golden、expected 和 breaking fixtures。v1 snapshot 冻结原始输出兼容；v2 snapshot 冻结 NVIDIA GPU 与华为昇腾 NPU 的显式类型，breaking fixture 证明 vendor-kind 错配会明确失败。

## Import and reproduce

```bash
python -m gpu_scheduler_lab import-mini-ai-cloud \
  --input tests/fixtures/mini_ai_cloud/v2-golden.json \
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
- 不把 v2 结果回写 Mini-AI-Cloud；文件合同是单向离线输入；
- 不共享 SQLAlchemy/Pydantic 内部类；
- 不把 simulator 的 vendor preference、fallback 或 calibrated profile 解释为真实硬件性能；
- 不替代 Mini-AI-Cloud 的 transaction、reservation、lease、execution fencing；
- 不声称模拟结果验证真实 Docker/Kubernetes/GPU runtime。
