# Alibaba 2026 spot-GPU trace

## Supported release

首个 production trace adapter 支持 Alibaba Cluster Trace Program 的 `cluster-trace-v2026-spot-gpu` CSV release：

- upstream：https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-spot-gpu
- node file：`node_info_df.csv`
- job file：`job_info_df.csv`
- adapter name：`alibaba-spot-gpu-v2026`

上游 README 说明 node table 覆盖 4,278 个 GPU Node 和 6 种 GPU 类型，job table 提供 organization、requested GPU model、CPU/GPU request、worker count、relative submit time、duration 与 HP/Spot 类型。该 release 附随 ASPLOS 2026 GFS 论文。

Alibaba 另有规模更大的 `cluster-trace-gpu-v2026` ASI hourly fact tables。它适合 cluster characterization，但 job execution summary 不直接提供本 adapter 所需的 arrival sequence；本 Phase 没有把两种 schema 混为一个格式。

## Attribution and use boundary

完整数据不提交到本仓库。上游项目说明数据用于 research/study，并请求使用者引用对应论文；上游目录未给出可由本项目替用户重新授权的独立数据许可证。下载和使用前应复核上游 README、论文 citation 与当前使用条款。

测试 fixture 只包含上游 README 公开展示的示例行，并在 fixture README 中标明来源。

## Download

默认下载到 `.data/`，该目录被 Git 忽略：

```bash
python scripts/download_trace.py \
  --output-dir .data/alibaba-spot-gpu-v2026
```

脚本只从 Alibaba 官方 GitHub repository 的 raw files 下载，并写入 `source-manifest.json`，记录 source URL、dataset name、下载时间和 byte size。已存在文件默认不覆盖。

## Import

```bash
python -m gpu_scheduler_lab trace-import \
  --format alibaba \
  --input .data/alibaba-spot-gpu-v2026 \
  --start 0 \
  --duration 86400 \
  --max-jobs 10000 \
  --max-nodes 1000 \
  --sample-rate 1.0 \
  --seed 42 \
  --output scenarios/alibaba-day1.generated.yaml
```

如果公开 GPU model 没有内置 memory capacity，必须显式提供：

```bash
--gpu-memory GPU-series-1=24
```

未知 model 默认报错，不能静默猜测。`--skip-invalid` 可以显式跳过坏 Job row；metadata 会保存 bounded warning、invalid count 和其他 normalization statistics。

## Normalization semantics

| Source field | Scenario field | Rule |
| --- | --- | --- |
| `node_name` | Node `id` | string identity |
| `gpu_model` | GPU `model` | exact public label |
| `gpu_capacity_num` | GPU inventory | 必须为正整数 |
| `job_name` | Job `id` | stable string |
| `submit_time` | `arrival_time` | window/filter 后减 selected minimum，归零 |
| `duration` | `duration` | positive seconds |
| `gpu_request * worker_num` | `gpu_count` | 两者必须为正整数，不支持 fractional GPU |
| requested `gpu_model` | Job `gpu_model` | exact constraint |
| GPU model capacity | `gpu_memory_gb` | 显式 mapping；metadata 记录这是 inference |
| HP / Spot | high / low | deterministic priority mapping |
| `organization` | `group` | fairness grouping |
| `cpu_request` | `source_metadata` | 保留但不参与 placement |

Filtering 顺序固定为 parse/validate、time window、seeded SHA256 sampling、`(submit_time, job_id)` sort、`max_jobs`、timestamp normalization。`max_nodes` 按 Node ID 稳定截取。

## Limitations

- spot-GPU CSV 不提供 rack/zone，adapter 不伪造 locality；
- CPU capacity/request 被保留为 metadata，但 Phase II 没有扩展通用 CPU scheduler；
- GPU memory demand 由 requested model capacity 推断，不是 trace 中观测的 per-Job memory request；
- duration 是 trace observation，不是生产调度器可准确预知的值；
- trace replay 不是 real GPU、network、checkpoint 或 control-plane validation。

## CI fixture

CI 只运行：

```bash
python -m gpu_scheduler_lab trace-import \
  --format alibaba \
  --input tests/fixtures/alibaba_trace_sample \
  --max-jobs 2 \
  --gpu-memory GPU-series-1=24 \
  --output build/alibaba-sample.yaml
```

完整 dataset 缺失不会影响 package import、unit tests 或 CI。
