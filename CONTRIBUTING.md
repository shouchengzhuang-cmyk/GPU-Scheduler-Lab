# Contributing

GPU Scheduler Lab 优先接受研究问题明确、行为可复现、证据与外推边界清楚的改动。

## Before you start

1. 搜索现有 Issue 和 PR；新策略、指标或场景应先说明研究问题、基线和非目标。
2. 从最新默认分支创建短生命周期分支，不在同一 PR 中混入无关重构、依赖更新或结果改写。
3. 不要提交生产 trace、租户标识、凭据或无法公开的数据。

## Local setup

项目以 Ubuntu/WSL 与 Python 3.12 为验证环境：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/mypy .
.venv/bin/pytest
```

## Research and behavior changes

- 对照策略必须使用相同 workload、seed、容量事件和 simulation engine。
- 修改调度、指标、事件顺序或场景语义时，增加针对性测试，并在代码冻结后重跑受影响 demo/benchmark。
- 报告 workload、样本量、commit、环境、统计口径与原始产物；不要只展示有利指标。
- 仿真和 synthetic trace 标记为 `SIMULATED`；仅真实外部组件实际运行可标为 `REAL`；未执行项写 `NOT RUN`。
- 不把 simulator wall time 外推为真实 scheduler、GPU、CUDA、网络或 Kubernetes 性能。

## Pull requests

使用 Conventional Commits，例如 `fix(reclaim): preserve active fleet capacity`。PR 必须说明变更、问题、非目标、验证、风险和证据边界；面向用户的变化还应更新 README、[CHANGELOG](CHANGELOG.md) 和相关设计文档。版本与兼容要求见 [release policy](docs/release-policy.md)。
