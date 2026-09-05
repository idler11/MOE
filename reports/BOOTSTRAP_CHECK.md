# 交接包发布前核对

2026-09-05，ChatGPT 在本会话的 CPU 容器运行了 scripts/recheck_legacy.py。实际结果见 bootstrap_container_20260905.json：进程退出码 0，legacy 四文件哈希不变，rtol=1e-6 / atol=1e-9 下无超容差差异。结果文件字节哈希与历史 results.json 不同并不意味着数值核对失败；判定基于逐字段比较，不强求浮点末位完全相同。

另对 wrapper 比较函数进行了 5 个正/负检查，并解析了两个工具脚本的语法。这些不是科学核心的 P1 测试。新的 src/moe_event 与 P1 tests 仍由 Codex 实现。

本次运行不在用户机器、不涉及 GPU，也不是预训练模型、速度或泛化证据；PROJECT_STATE 仍将用户 P0/P1 标为待执行。原始包是 read-only reference，不能让复现覆盖原结果。用户本机需重新执行并提交自己的环境与证据。
