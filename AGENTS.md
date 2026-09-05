# Project instructions: MoE finite-step event correction

## 项目与目标

本仓库只研究单卡 RTX 4090 条件下的 MoE 量化：在相同合法量化候选集合上，有限步路由事件校正是否减少候选选择遗憾，并最终改善真实模型质量/成本。不是 TraceMem、Agent Memory 或通用 RL 项目；不得从其他仓库导入要求、数据或叙事。

科学主张尚未成立；新颖性审查尚未通过。不要写“首次”“SOTA”“已证明真实模型有效”。数学恒等式、已有线性代数和 Taylor 余项不独立构成原创贡献。

## 每次会话的读取顺序

1. README.md 和 PROJECT_STATE.json。
2. START_PROMPT.md 与 DECISIONS.md。
3. docs/01_THEORY.md、docs/02_EXPERIMENTS.md。
4. docs/03_ROADMAP.md、docs/04_ENGINEERING.md。
5. docs/05_RELATED_WORK.md、docs/06_COLLABORATION.md。
6. 当前任务 Issue、已有 reports 与待审查 diff。

先读取真实文件与 git status，不依赖上一次会话记忆。远程仓库应为 idler11/MOE；发现不一致时不写入。不要在错误目录继续执行。

## 首轮授权

现在只执行 P0 环境/遗留包复现与 P1 独立数学核心/测试。两者均以 CPU 为准。完成后提交证据并停止，等待研究审查，不自行跑 P2-P6。后续阶段必须在 PROJECT_STATE.json 记录明确授权与对应证据；Codex 不自行批准自己的科学结论。

不自动下载大模型、不执行付费 API、不启动多 GPU/大训练、不重装系统 CUDA、不替换已有工作的 PyTorch GPU 安装。缺依赖先诊断，必要时在项目专用环境中按实际平台安装；不把 CPU wheel 覆盖到用户 GPU 环境。不要配置未经确认可用的模型名称。

## 不可破坏的科学约束

- 主对象是有限、可部署的量化码修改，不是可以独立控制每个 token 的 oracle 路由。
- 有限经验目标在不跨界邻域内的普通分支梯度可以正确；禁止重用“自动微分必然漏梯度”的已撤回叙事。
- 事件项在候选新输入上比较旧集合与新集合，保留符号及完整门控聚合；不能替换成非负专家距离。
- 固定分支并不固定 gate 数值。两种 top-k 概率归一化方式都要测试；真实模型保持原实现。
- 保持路由集合的零空间不是功能不变的零空间。路由几何的精确条件仅限文档假设；偏置、分组路由、capacity drop、非线性上游必须标记不支持或另做推导。
- 当前量化点的梯度一般非零；完整 Hessian 不等于 Gauss-Newton。所有近似必须有名称与误差记录。
- 测试代码可以使用真实候选损失作为 oracle；算法选择接口不得读取留出集或隐藏 oracle 表。
- 不使用提升指标作为单元测试必过条件。零收益、负收益、无路由事件、二阶模型已经选对，都是必须保留的合法结果。
- 不允许看到 test 结果后换种子、候选或阈值来保住假设；修改方案必须版本化，并使用新的封存验证。

## 证据与数据

legacy/moe_theory_audit 下四个文件保持字节不变，哈希见 legacy/MANIFEST.json。原始脚本会覆盖同目录 results.json，因此只通过 scripts/recheck_legacy.py 在隔离副本运行。历史结果是随机 toy 证据，不是本机复现或预训练模型结果。

默认实验数据由本项目程序生成，按 seed、生成器版本和完整序列切分；不导入其他项目数据。公开标准 benchmark 或外部语料的使用须另行明确批准。公开模型权重不等于实验数据，仍须在 P3 获批后固定 revision、配置、许可证与本地缓存路径。

每次运行记录 run_id、代码 SHA/dirty 状态、配置与数据哈希、seed、命令、退出码、实际依赖、dtype、设备、数值容差、耗时与资源。原始大日志/权重/激活放在被忽略的 artifacts、data、cache、models 下；提交可审查的聚合 JSON/CSV、失败情况与复现命令。公开仓库严禁密钥、私人文本或完整环境变量转储。

## 协作与 Git

保护用户未提交的改动；不 force push、不 reset --hard、不删除历史。使用 codex/p0-p1-math-core 等工作分支，小步提交，提供 PR 或 commit 供 ChatGPT 审查。不要自动合并自己的研究实现。

ChatGPT 负责理论/文献/实验决策审查，Codex 负责实现/运行/如实反馈，用户提供执行环境并决定下一阶段。每轮必须更新 reports 与 PROJECT_STATE.json 的事实状态，区分 implementation_complete、tests_passed、review_approved；三者不能互相替代。

AGENTS.md 是持续约束，START_PROMPT.md 是本轮入口，路线图未来命令并不意味着对应脚本已经存在。
