# Security policy

## Supported versions

安全修复优先面向默认分支和最新发布版本。历史标签用于复现实验，不承诺持续回补。GPU Scheduler Lab 是离散事件研究工具，不应处理生产凭据、真实租户数据或直接控制集群资源。

## Report a vulnerability

请通过仓库 Security 页面中的 [private vulnerability reporting](https://github.com/shouchengzhuang-cmyk/GPU-Scheduler-Lab/security/advisories/new) 私下报告。不要公开提交未修复漏洞，也不要附带真实 token、私钥、Cookie、生产 trace 或可识别个人/租户的数据。

报告请尽量包含：

- 受影响的版本或 commit SHA；
- 最小复现步骤和所需输入；
- 影响范围、攻击前提和已知缓解方式；
- 已脱敏日志、构造的最小 fixture 或测试用例。

维护者会尽力确认、修复并协调披露，但不承诺固定响应 SLA。一般正确性或结果解释问题请使用普通 Issue。
