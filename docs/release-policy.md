# Version, release and deprecation policy

## Versioning

GPU Scheduler Lab 使用语义化版本管理公开 Python API、CLI、scenario schema、trace contract 和指标语义：

- `MAJOR`：需要迁移的破坏性变化；
- `MINOR`：向后兼容的策略、指标或工具扩展；
- `PATCH`：不破坏契约的正确性、安全或文档修复。

开发快照使用 PEP 440 形式，例如 `0.3.0.dev0`。`0.x` 阶段仍需对公开契约执行弃用和迁移流程。

## Release gate

发布前至少满足：

1. package metadata、`gpu_scheduler_lab.__version__`、CLI 和文档版本一致；
2. wheel 可构建，并能在独立虚拟环境安装和运行 `gpu-scheduler-lab --help`；
3. Ruff、mypy、pytest 及仓库 CI smoke 全部通过；
4. 受影响的策略、指标或场景已在冻结 commit 上重跑，产物记录输入、seed 与环境；
5. CHANGELOG、风险、限制和迁移说明完整，且没有提交敏感数据；
6. release tag 和产物只从已确认的默认分支 commit 生成。

commit、push、PR、merge、tag、release 和部署是不同动作，任一步通过都不自动授权下一步。

## Deprecation

替换公开 CLI、Python API、scenario 字段或指标时，先提供新入口、迁移说明和兼容测试。旧入口至少保留一个已公告的开发或发布周期，并在运行时或文档中给出明确警告；计划移除版本写入 CHANGELOG。安全问题需要提前移除时，必须解释风险和替代方案。
