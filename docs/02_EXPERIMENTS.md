# 02｜实验设计与否证标准

## 1. 分开回答三个问题

A. 评价仪器是否正确？P0/P1，用精确数学与独立实现。

B. 合法候选中是否存在可获得收益，事件校正是否改善选择？P2/P4。

C. 同等总校准成本与部署预算下是否值得，完整模型是否更好？P5/P6。

任何一个问题的通过都不能代替另一个。当前仅有 legacy 随机 toy 文件，不存在预训练实验或速度证据。

## 2. 比较对象与共同候选

首先冻结候选生成器：固定尺度、整数范围、分组与位宽，Q0 为最近舍入结果，按距舍入中点距离选 p 个合法坐标（稳定排序破 tie），枚举 2^p 个另一侧舍入/保持选择。clipping 后重复候选去重并记录；Q0 保留。

v0 的 p=8 是局部搜索大小而非模型低秩假设。每个方法获得同一个候选集合和同一份 calibration 数据。选择机制与 evaluation oracle 物理分开，防止候选真实损失泄漏。

必要对照：

- no-op / nearest rounding。
- 当前分支一阶、完整二阶；GN/diagonal 分别命名。
- 完整二阶 + 精确带符号 event 项（机制上限版本）。
- 完整二阶 + 非负 jump²（验证交叉项是否必要）。
- 路由变化数或 margin 惩罚（其超参仅在 development 调）。
- exact fixed-branch loss 与 exact candidate loss（诊断 oracle，报告实际成本）。
- 同总预算的直接候选子集搜索、更多普通校准样本、现有 PTQ/路由方法。

随机候选子集基线与主方法须共享候选生成母集合，固定 subset seed。诊断 oracle 可评估全部候选；生产方法不能免费读取 oracle 结果。

## 3. 指标与分母

主指标：V=L(Q0)-min_C L，R_method=L(Q_selected)-min_C L，方法改进 R_base-R_event，以及所选同一个 Q 在独立 holdout 的损失差。

辅助：proxy MAE、ranking/Spearman、top-1 agreement、top-r regret、事件比例、负事件项比例、gate 模式、near-tie 比例、layer/bit/magnitude 分层。Spearman 对常量数组不可定义，输出 null 与 reason，禁止 NaN 混入 JSON。

事件率按专家集合变化定义，不因集合顺序重排误计。未事件 token 的事件项应为零（浮点容差内）。V≈0 的案例必须保留，normalized 指标为 null 并注明 denominator，不把零机会 case 删除。

全模型主指标在 P6 前预注册：例如生成文本的 heldout NLL；公开 benchmark 需另获批准。局部重构不能替代最终性能。

## 4. 数据政策与独立性

所有默认实验数据由本仓库程序生成。toy 使用 RNG 模型与独立 calibration/holdout 输入。真实模型文本优先由无付费 API 的程序模板生成，包含叙述、代码样式、逻辑/算术等多种分布；记录生成器版本、family、seed 与 text/token 哈希。

它们不是自然语言分布的代表性证明。需要外部标准语料或 benchmark 才能建立的结论必须写为未测，并在获批后单列外部数据实验。

按完整序列/模板族切分，不把同一句的 token 分到两边。候选选择、超参、阈值不得访问封存集。固定 teacher、tokenizer 与 reference precision；量化点 Q0 与 teacher 可能不同，不假定 gradient=0。

## 5. P2 预注册建议（目前不可执行）

通过 P1 后，由研究负责人冻结配置再运行：development seeds 0-9，confirmation seeds 100-129；同 seed 的两个 gate 模式作为配对子实验，不视为独立重复。输入种子域与模型种子域分开。先 p=8、3-bit、256 calibration / 512 holdout token 为默认起点，扩展 2/4-bit 与 p=4/12 时先冻结预算。

这些是设计值，不是已运行的结果或样本量充分性保证。legacy seeds 0/1/2 只能作为 development，不进入最终确认。

独立统计单位为模型 seed（toy）或完整上下文/预先定义 block（真实模型），不是 token×candidate 的所有格子。报告配对原始差、bootstrap 区间和各 seed 结果；统计与 practical minimum 需运行前固定。置信区间跨零意味着不确定，不自动推出“机制不存在”。

P2 的目的不是筛选几颗获胜 seed，而是识别机制边界：事件密度、分支差异、量化步长、二阶余项与选择改进之间的关系。

## 6. 数值控制

数学参考使用 CPU float64；实际前向检查 float32/BF16 时各自报告数值误差底噪。near-tie 候选仍进入总表，但标记事件不稳定区间，分层展示。重复同一运行估计噪声，不根据观察到的收益事后放宽“通过阈值”。

先测恒等式与 actual branch parity，再测科学效果。失败时输出最小 case 的 h、logits、sets、residual/jump（只在自生成 toy 可公开），而非改 expected value。

## 7. 成本账本

所有方法统计 candidate construction、full reference、g/H 构造、router screening、endpoint experts、exact shortlist、validation、数据搬运的时间/次数。公共开销与独有开销分别列出，但同总预算比较必须包含全部必要开销。

CPU 用单调时钟；GPU 正确 synchronize、warmup，并记录 max_memory_allocated/reserved、宿主 RAM、缓存命中率与 dtype。冻结硬件/线程/上下文长度。稀疏 expert call 更少不自动代表实际更快。

部署时保持同位宽与格式；比较 average bits 时包括 scales/zero-points/未量化部分的存储。fake-quant 框架没有 low-bit 推理速度声明资格。

## 8. 必须允许的停止结论

- V 或 R_base 相对数值噪声/实际需求太小：没有足够决策空间。
- 事件纠偏只提高相关性，不改善选择：不满足主目标。
- 只在 calibration 获益：可能过拟合。
- 同算力 direct search / 更多校准更好：没有计算经济性。
- 高精度保留少量 attention/router 就解决：复杂方法价值不足。
- 效果只在局部 W_O 存在，无法影响重要压缩对象：实用范围有限。
- OPERA/Q-Strata/其他直接工作覆盖了主机制：先修订新颖性，不靠改名继续。

停止、修订与扩大实验都要通过报告和 DECISIONS 留痕，不能自动通过。
