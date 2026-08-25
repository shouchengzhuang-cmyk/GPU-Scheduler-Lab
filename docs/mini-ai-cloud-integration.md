# Mini-AI-Cloud integration contract

## Boundary

联合采用 `mini-ai-cloud.gpu-scheduler-lab/v1` JSON 文件契约。GPU Scheduler Lab 不连接 Mini-AI-Cloud 数据库/API，不导入其内部包，也不修改其调度状态。这样可独立复现 benchmark，并避免两个项目的 schema migration 或服务依赖互相锁定。

Mini-AI-Cloud 当前概念到 simulator 的映射：

| Mini-AI-Cloud | GPU Scheduler Lab | 说明 |
|---|---|---|
| Worker `id` | Node `id` | Worker 作为资源/故障域 |
| Worker schedulable state | Node `schedulable` | false 时保留 physical inventory，但不计入可调度指标分母 |
| Worker labels | Node topology | v1 保留 metadata；Phase II 可读取 `zone`/`rack`，但不要求主项目改契约 |
| GPU `device_uuid` | GPU `id` | 保持具体设备 identity |
| `memory_total_mb` | `memory_capacity_gb` | 除以 1024；v1 未提供 model 时使用 `generic` |
| unhealthy GPU | filtered | 不进入可调度 inventory |
| Task `queued_at` / `arrival_time` | Job `arrival_time` | ISO timestamp 归一化到最早任务为 0 |
| `duration_seconds` / `timeout_seconds` | Job `duration` | 前者优先；这是实验输入，不是运行时预测 |
| `gpu_count` | Job `gpu_count` | CPU-only Task 会被过滤并记录数量 |
| `gpu_memory_mb` | Job `gpu_memory_gb` | 必须为正数 |
| priority 0–100 | low/normal/high/critical | 0–24 / 25–74 / 75–89 / 90–100 |
| `project_id` | Job `group` | 用于按完成率与延迟效率计算 Jain fairness |
| workload type | inference/training | `training`/`batch_job` 映射 training，其余映射 inference |
| label `gpu_scheduler_lab/gang=true` | `gang=true` | 多 GPU Task 也默认 gang |

## Input example

完整示例见 `scenarios/mini_ai_cloud_demo.json`：

```json
{
  "contract_version": "mini-ai-cloud.gpu-scheduler-lab/v1",
  "workers": [{
    "id": "worker-a",
    "schedulable": true,
    "gpu_devices": [{
      "device_uuid": "GPU-a0",
      "memory_total_mb": 24576,
      "health": "healthy"
    }]
  }],
  "tasks": [{
    "id": "task-a",
    "arrival_time": 0,
    "duration_seconds": 60,
    "gpu_count": 1,
    "gpu_memory_mb": 20480,
    "priority": 50,
    "project_id": "project-a",
    "workload_type": "batch_job"
  }]
}
```

转换命令：

```bash
python -m gpu_scheduler_lab import-mini-ai-cloud \
  --input scenarios/mini_ai_cloud_demo.json \
  --output scenarios/mini_ai_cloud_demo.generated.yaml
```

生成 YAML 的 `metadata` 保留 source、contract version 和过滤的 CPU-only Task 数量。

Phase II 没有擅自升级文件 contract。若未来 v2 增加 GPU model 或 zone/rack，adapter 必须继续接受 v1，并且仍只做离线文件转换。

## Result handoff

`benchmark` / `compare` JSON 包含 scheduler、完整 metrics、逐 Job outcome 和 trace。Mini-AI-Cloud 可把它作为离线 policy study artifact 保存，但不得把 simulation completion、SLA 或 utilization 当成真实 Task/Worker 运行结果。

## Non-goals

- 不从生产数据库直接导出或回写；
- 不共享 SQLAlchemy/Pydantic 内部类；
- 不替代 Mini-AI-Cloud 的 transaction、reservation、lease、execution fencing；
- 不声称模拟结果验证真实 Docker/Kubernetes/GPU runtime。
