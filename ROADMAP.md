# Roadmap

本路线图描述研究与工程优先级，不承诺交付日期或生产支持。结果声明始终以仓库当前 commit、测试和实验产物为准。

## Current: stable project contract

- 统一 distribution、CLI、运行时版本和发布元数据。
- 建立 wheel 安装冒烟、贡献、安全、变更记录与发布规则。
- 在 README 首屏固定研究问题和 simulator 外推边界，不改变现有策略或场景行为。

## Next: canonical experiments

- 为关键策略比较冻结可审计的 scenario、seed、指标、环境和输出格式。
- 同时报告 utilization、wait、SLA、fairness、fragmentation 和 reclaim/overhead，不按结果挑选指标。
- 追踪 synthetic、公开 trace fixture 与未运行真实环境之间的证据差距。

## Later: validated extensions

- 只有在研究问题和基线明确后才引入新的策略、指标或 trace adapter。
- 对大规模仿真建立复杂度与性能回归基线，但不外推为真实控制面吞吐。
- 若未来接入真实 GPU/Kubernetes，必须以独立 adapter、隔离环境和 `REAL` 证据推进，不能改写既有模拟结果的性质。

候选工作通过 Issue 和独立 PR 进入；路线图不等于能力已实现或验证通过。
