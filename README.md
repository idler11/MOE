# MoE Event-Corrected Quantization

**研究工作台：有限步路由事件校正的离散量化决策。**

本仓库由用户创建。研究目标是在单张 RTX 4090 约束下，验证：在同一组合法量化候选里，加入带符号的终点路由事件项，能否减少普通固定分支局部模型的候选选择遗憾。

> 当前状态：研究交接与数学审查阶段。没有预训练 MoE 实验结果，没有量化加速结果，没有通过完整新颖性审查。随机 toy 结果不是论文结论。

## 核心对象

对当前量化矩阵 Q0 和合法候选 Q1：

`真实有限步损失变化 = 固定旧专家集合的损失变化 + 在新输入处计算的路由事件项`

方法草案使用低维候选坐标中的一阶/二阶模型近似第一项，只对专家集合发生变化的 token 计算第二项。最终选择可部署的量化码，不能独立为每个 token 指定最佳专家。

## 从这里开始

| 文件 | 用途 |
|---|---|
| [START_PROMPT.md](START_PROMPT.md) | 直接交给 Codex 的首轮任务 |
| [AGENTS.md](AGENTS.md) | 永久研究边界与执行纪律 |
| [docs/01_THEORY.md](docs/01_THEORY.md) | 完整数学对象、推导、假设与反例 |
| [docs/02_EXPERIMENTS.md](docs/02_EXPERIMENTS.md) | 候选定义、对照、指标、数据切分与停止条件 |
| [docs/03_ROADMAP.md](docs/03_ROADMAP.md) | 每阶段输入、工作、产物、验收与负责人 |
| [docs/04_ENGINEERING.md](docs/04_ENGINEERING.md) | 模块接口、测试清单与资源约束 |
| [docs/05_RELATED_WORK.md](docs/05_RELATED_WORK.md) | 已核对来源、待查重项与禁止的创新声明 |
| [docs/06_COLLABORATION.md](docs/06_COLLABORATION.md) | ChatGPT / Codex / 用户协作闭环 |
| [PROJECT_STATE.json](PROJECT_STATE.json) | 当前事实状态与下一步授权范围 |
| [DECISIONS.md](DECISIONS.md) | 决策与被替代的旧说法 |
| [legacy/README.md](legacy/README.md) | 原始数学审查包及不可变哈希 |

## 第一轮只做 P0 / P1

先核对环境、复现历史审查包，再由 Codex 实现独立、可测试的 toy 核心。不要直接下载 OLMoE、训练模型、做大规模 sweep 或升级系统 CUDA/PyTorch。

已有交接工具：

```text
python scripts/validate_handoff.py
python scripts/recheck_legacy.py --output artifacts/p0_recheck_001
```

第二条需要当前 Python 环境已有 NumPy、SciPy、PyTorch。缺依赖时先报告；不要为此替换已工作的 GPU 环境。历史脚本会覆盖同目录 results.json，因此只能通过隔离复现工具执行。

实验算法代码、模块化测试与模型适配器留给 Codex 实现；路线图中的未来命令不是已经存在的程序。

## 协作方式

ChatGPT 审查理论、文献和实验决策；Codex 写代码、执行并提交证据；用户提供本地环境并决定是否进入下一阶段。首轮完成后，把工作分支/提交与 reports/P0_environment.md、reports/P1_math.md 交回审查，不自动开始 P2。

本仓库当前公开。不要提交密钥、私人文本、模型权重、激活缓存或其他项目材料。开源许可证尚未由用户选择；不得冒称已获许可转载论文或第三方代码。
