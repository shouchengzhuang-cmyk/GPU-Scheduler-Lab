# GPU Scheduler Lab 0.4.0 release checklist

本清单只准备 `0.4.0` release candidate。完成勾选不自动授权 tag、GitHub Release 或部署。

## Identity and contracts

- [ ] `pyproject.toml`、`gpu_scheduler_lab.__version__`、README 和 wheel metadata 都是 `0.4.0`。
- [ ] Mini AI Cloud 输入 v1/v2 与 result handoff v1 schema 可解析。
- [ ] v1/v2 golden fixture 完整导入，breaking fixture 明确失败。
- [ ] 输出 JSON 固定标记 `evidence_kind: SIMULATED`。
- [ ] CHANGELOG 只记录当前 Git 历史和生成产物可证实的事实。

## Quality and reproducibility

```bash
ruff format --check .
ruff check .
mypy .
pytest
make reproduce-study
```

- [ ] 五条命令均在准备创建 release 的同一 commit 上通过。
- [ ] `build/study/canonical/manifest.json` 的 `git.sha` 与该 commit 一致，`dirty_tree` 为 false。
- [ ] `python -m gpu_scheduler_lab study verify --input build/study/canonical` 通过。
- [ ] 正式报告的表格、图、Markdown 和 `hashes.sha256` 来自同一 summary。
- [ ] wheel 在隔离虚拟环境安装，`gpu-scheduler-lab --help` 和 package version 检查通过。

## Evidence boundary and publication

- [ ] 报告、README 与 PR 明确这是离散事件 simulation，不是真实 GPU/Kubernetes/生产 scheduler 证据。
- [ ] 未提交完整公开 trace、凭据、用户数据、数据库转储或本地路径中的敏感信息。
- [ ] PR CI 全绿且依赖 PR 已合入默认分支。
- [ ] release commit 已从默认分支重新核对，不从未合并功能分支发布。
- [ ] 用户单独授权后才创建 tag 或 GitHub Release；部署需要再次单独授权。
