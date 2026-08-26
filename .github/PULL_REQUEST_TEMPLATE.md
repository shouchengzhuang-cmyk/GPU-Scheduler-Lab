## 变更内容

- <!-- 请概括实际改动。 -->

## 解决的问题

- <!-- 请说明研究问题或工程问题，以及解决方式。 -->

## 非目标

- 本 PR 明确不处理：

## 验证方式

- [ ] `.venv/bin/ruff format --check .`
- [ ] `.venv/bin/ruff check .`
- [ ] `.venv/bin/mypy .`
- [ ] `.venv/bin/pytest`
- [ ] 受影响的 demo/benchmark 已在冻结 commit 上重跑
- [ ] 未执行项已标记为 `NOT RUN` 并说明原因

## 风险与注意事项

- 影响的策略、事件顺序、scenario schema、指标、结果或发布流程：
- 回滚方式：
- 如无明显风险，请写“暂无明显风险”。

## 证据

- 证据标签：`REAL` / `SIMULATED` / `NOT RUN`
- commit、环境、workload、seed 与产物：
- 指标口径、对照基线和未覆盖范围：

## Checklist

- [ ] 改动范围单一，没有夹带无关重构、依赖或配置变更
- [ ] 行为变化包含针对性测试，且结果来自本次冻结 commit
- [ ] 文档和 CHANGELOG 已更新，没有夸大 simulator 结论
- [ ] 未提交凭据、生产 trace、租户标识或其他敏感数据
- [ ] 破坏性变化包含迁移、弃用和兼容计划
