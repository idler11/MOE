# 05｜文献与创新边界

记录日期：2026-09-05。此文件是排重工作单，不是已通过 novelty review。只有读到的材料才支持描述；搜索摘要不代表全文核查。未来实验前再次检查新版本。

## 本轮可核对的直接来源

| 工作 | 来源与核对深度 | 对本项目的约束 |
|---|---|---|
| VSRAQ | https://arxiv.org/abs/2606.05688 ，摘要 | 已以数值/结构对齐保护路由；不能声称首次保护 Top-k |
| GEMQ | https://arxiv.org/abs/2605.23078 ，摘要；作者代码 https://github.com/jndeng/GEMQ | 已处理全局 bit allocation、router 调整、渐进量化；全文中的近似条件仍需逐项读 |
| SRA-MoE / DREAM-MoE | https://blog.nota.ai/insights/icml-2026-workshop-moe-quantization ，作者方法说明 | 已有选择性输出感知路由对齐和下游路由监督；作者说明是 ICML workshop，不当成主会证据 |
| OPERA-MoE | https://sites.google.com/view/aihalab/publications ，作者发表列表 | 标题为 Output-aware Routing Perturbation Alignment via Observable Discrepancy for MoE Quantization；完整方法尚待取得，高重叠风险 |
| Q-Strata | https://arxiv.org/abs/2608.30564 ，搜索检索的论文摘要，2026-08-31 | 已有离散配置候选与组装模型目标的双层分配；“直接比较量化候选”本身不能算新。尚未全文核对 |
| 非可微模型的边界梯度 | https://arxiv.org/abs/1806.00176 ，此前讨论的追溯线索，本轮未全文核查 | 一般区域/边界分解不是本项目原创，需要准确引用原文 |
| MoE 边界几何 | https://arxiv.org/abs/2606.19036 ，此前讨论的追溯线索，本轮未全文核查 | 不宣称首次发现 Top-k 不连续或多面体路由几何 |

没有将论文正文拷贝进公开仓库，也没有编造不曾读到的页码/实验结果。

## 模型与工具的官方来源

OLMoE 配置：https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/main/config.json 。本轮看到的配置供选型，执行时必须固定 revision 并核查源码；legacy 的 top-k renorm 不可直接当真实模型聚合。

OLMoE 论文：https://arxiv.org/abs/2409.02060 。引用前核查需要支持的具体主张。

Codex AGENTS 指南：https://developers.openai.com/codex/guides/agents-md （本轮重定向到官方 ChatGPT Learn 文档）。根目录 AGENTS.md 提供持续指令；START_PROMPT.md 仍需显式让执行者读取，不能假设所有长文档自动装入上下文。

## 拟争取的差异：必须验证，不是现成结论

在同一组共享、合法、可部署的量化候选上，定位固定分支代理的选择错误；用候选终点的带符号事件项纠正评分；在相同总校准预算下获得真实模型收益。

分解恒等式、Taylor 余项、低维 router 对比空间、regret<=2epsilon 都是已有数学基础。贡献必须是机制重要性 + 有效可计算估计 + 真实收益/解释，不是为基本代数换名字。

## 全文排重需填写的表

每篇记录：确切标题、版本/提交日期、是否主会/Workshop/预印本、优化变量、候选集合、是否计算替代 expert/终点分支、事件损失是否有符号、是否把旧路由作标签、全局/局部目标、稀疏计算策略、成本是否计入、代码入口、与本项目重叠处。

先处理 OPERA、GEMQ、SRA/DREAM、VSRAQ、Q-Strata，再补 GPTQ/rounding/search/非光滑优化直接基线。查到重叠先修改研究命题或停止，不可仅改名称。

## 禁止的写法

“文献没人研究路由错误”“首次功能感知”“Hessian 永远无法处理 MoE”“新颖性因为没搜到关键词所以通过”“toy 选中 oracle 证明实际最优”“4090 可跑所以能发顶会”。
