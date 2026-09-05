# 历史数学审查包（只读证据）

moe_theory_audit 中四文件来自本会话提供的 moe_theory_audit.zip，内容保持 byte-for-byte 不变。SHA-256 与原 archive 哈希见 MANIFEST.json。

不要直接执行原目录的 verify_math.py：它会覆盖原 results.json。使用根目录 scripts/recheck_legacy.py，在新的 artifacts 子目录生成隔离副本并对比。根目录 .gitattributes 固定 LF，避免 Windows checkout 改换行导致哈希失真。

原 README 的 pip install 示例是历史内容，不是本项目自动安装指令；不得照此覆盖用户已有 GPU PyTorch 环境。

这些结果来自随机、小型、CPU/float64、密集 expert 计算的 toy。没有预训练 MoE、留出泛化、低比特内核或速度证据。文件名 results 不意味着研究成果已成立。复现与代码模块化是 Codex 首轮任务。
