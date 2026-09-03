# Changelog

本项目的重要变更记录在此文件中。格式参考 Keep a Changelog，版本遵循 [release policy](docs/release-policy.md) 中的语义化版本规则。

## [Unreleased]

## [0.4.0] - 2026-09-03

### Added

- Typed Mini AI Cloud v2 accelerator exchange: explicit NVIDIA GPU and Huawei Ascend NPU constraints, runtime profiles and capabilities.
- Heterogeneous topology/outage simulation coverage, priority-safe cross-vendor reclaim, and contract timestamp regressions.
- Optional bounded parallel study execution with deterministic aggregation and resumable parallel outcomes.

### Changed

- Parallel study work is transplanted from the reviewed #15 series onto the current main baseline; the default remains serial.
- v2 timestamp schema and adapter accept documented UTC-naive values while v1 keeps its frozen compatibility boundary.

### Security

- Typed accelerator exchange remains offline and `SIMULATED`; no real GPU, Kubernetes, production scheduler, or deployment claim is introduced.

## [0.3.0] - 2026-08-26

### Added

- wheel 构建与独立 CLI 安装冒烟门禁。
- 许可证、安全策略、贡献指南、路线图和 GitHub 协作模板。
- 可恢复的 sensitivity/ablation runner、正式 study report、表格、图表与哈希清单。
- 12 条调度不变量合同、语义断言与 deterministic golden baseline。
- Mini AI Cloud v1 输入 JSON Schema、golden compatibility fixture 和 `SIMULATED` 结果 handoff schema。

### Changed

- 包、运行时、CLI、README 与 release checklist 统一为 `0.3.0`。
- README 首屏明确研究问题、模拟边界和项目治理入口。
- v1 adapter 明确过滤 CPU-only Task 与 unhealthy GPU，映射 model/topology，并接受、审计但不解释未知字段。

### Security

- 将开发测试依赖提升到 `pytest>=9.0.3,<10`，避免受 CVE-2025-71176 / GHSA-6w46-j5rx-g56g 影响的旧版本进入发布验证环境。

`0.3.0` 条目是 release 准备事实；尚未自动创建 tag、GitHub Release 或部署。
