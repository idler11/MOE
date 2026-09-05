# 03｜实施路线：每阶段的输入、工作、产物与验收

## 角色

ChatGPT：维护研究命题、数学正确性、文献差异、实验优先级；读取 Codex 提交和日志后给出继续/修正/停止判断。
Codex：实现、测试、运行、保存原始证据；报告实际发现而非迎合假设。
用户：启动本地执行、提供真实硬件环境、决定涉及数据/下载/后续计算的授权。

当前只授权 P0/P1。后续命令是接口目标，并非已存在程序。任何阶段都不得用“代码写完”代替“验收通过”。

## P0｜环境、来源与原始证据复现

输入：legacy 四文件、MANIFEST、当前主机。
Codex 工作：检查 remote/status/Python/依赖；执行 validate_handoff；隔离复制并执行原 verify_math；记录版本、命令、退出码、stdout/stderr、results_diff；检查原文件哈希不变。
现有命令：python scripts/validate_handoff.py；python scripts/recheck_legacy.py --output artifacts/p0_<unique_id>。
产物：reports/P0_environment.md，小型比较摘要，PROJECT_STATE 事实更新。
验收：原文件不变；实际执行成功或清晰阻塞；数值差异按预设容差审查。只有本机复现可声明本机 P0 完成。
ChatGPT 审查：确认不是照抄历史 numbers、不是环境被偷偷替换。
禁止：模型下载、GPU 大作业、重装系统环境。

## P1｜独立数学核心与测试

输入：docs/01_THEORY、P0 输出。
Codex 工作：src/moe_event 下的 reference/core 模块；typed result；两种 gate 模式；合法候选；手算和独立实现交叉验证；event-only 与 dense parity；pytest 与一个可执行 smoke 入口。
计划命令（需先实现）：python -m pytest -q；python scripts/run_p1_checks.py --config configs/p1_smoke.json --output artifacts/p1_<id>。
产物：实现、测试、小型 results JSON、reports/P1_math.md、可审查 commit/PR。
验收：工程规范里的必需测试逐一通过；无法支持的情形显式报错；不要求方法获胜；原始四文件哈希不变。
ChatGPT 审查：分支/门控语义、shared-weight 候选、teacher/reference、损失归一化与测试独立性。完成后停下，不自动 P2。

## P2｜toy 选择遗憾与留出泛化

前提：P1 经审查；先冻结 seeds/config/metrics/budgets 与确认集策略。
Codex 工作：同候选集下完整对照；按 seed 报告机会 V、选择遗憾、holdout 损失、事件密度；记录零/负收益与统计区间；机制消融而不是只挑好看图。
计划入口：scripts/run_p2_toy.py。
产物：configs/p2_preregistered.json、reports/P2_toy.md、长格式 candidate/selection 表、小型可审查 summary。
验收：没有 oracle/test leakage；确认集独立；是否存在有实际量级的选择改进由数据判断。结果不确定则报告不确定。
ChatGPT 审查：决定是否值得真实模型实验；同时完成更深的文献对照，不把 toy 通过当成新颖性通过。

## P3｜真实 MoE 适配与架构等价

前提：P2 审查、用户批准模型资源。首选候选 OLMoE-1B-7B-0924，不是已下载/已验证配置。
Codex 工作：固定 model/tokenizer revision，核对实际源码与 config；用极小输入检查自定义 local replay 与官方 forward 的 logits、Top-k、gate、expert 聚合和 residual parity；确认参数确实影响目标 router；测峰值显存与 host RAM。
计划入口：scripts/run_p3_adapter_parity.py。
产物：model_manifest.json（无权重）、数据生成 manifest、reports/P3_adapter.md。
验收：与所固定源码在明确容差内一致；tie 区间单列；不偷偷开启 top-k renorm；单 4090 预算可控。失败不进入效果实验。
ChatGPT 审查：精确几何假设是否满足、模型 replay 是否改变任务。按层/窗口运行，不做全模型 Hessian。

## P4｜真实合法候选上的机会与可修复决策错误

前提：P3 parity，通过批准的 generated data 与固定候选方案。
Codex 工作：预选若干层/投影，小候选集合 exact enumeration 得到诊断 oracle；计算 V、R_base、R_corrected、holdout transfer；比较 margin、负/正事件、direct search。
计划入口：scripts/run_p4_real_candidates.py。
产物：reports/P4_opportunity.md、layer/bit/context 配对表、失败案例。
验收问题：在不挑 seed/层的条件下，是否有显著超出数值噪声且有实际意义的基线选择遗憾？事件项是否能减少它？不是要求固定结论。
ChatGPT 决策：无机会则停止或缩窄机制论文，不能直接做复杂 optimizer。

## P5｜等预算的可部署校准器

前提：P4 有可利用机会，并记录先验可接受成本。
Codex 工作：sparse endpoint evaluator、g/H reuse、shortlist exact acceptance/no-op、有限步坐标搜索；包括同算力候选评估和更多普通校准强基线；验证 exported integer codes 与实际 forward 一致。
计划入口：scripts/run_p5_calibration.py。
产物：reports/P5_cost_quality.md、实测成本账本、对照配置与量化码哈希（大权重不进 Git）。
验收：同位宽/总存储/校准计算量下存在质量-成本价值；Hessian 和反事实计算开销不能隐藏；适用范围超过一个无关紧要的投影或如实限制。
ChatGPT 审查：是否被“保留少量 attention/router 高精度”击败；是否需要改为仅机制论文。

## P6｜完整模型、扩展与投稿证据

前提：P5 有效，直接相关文献查重通过或清楚区分。
Codex 工作：独立序列/生成族的全模型 NLL，获批准后的标准评测，第二配置/模型或跨层复验，seed/bit/cost 消融；fake quant 与 packed kernel 结论分开。
产物：reports/P6_full_model.md、paper/claim_evidence.md、全部图表对应的 runs/configs、失败与局限。
验收：每条论文主张有代码、原始证据、假设范围、比较对象与文献差异。选择会议需另核对当时征稿，不保证录用。
ChatGPT 工作：组织论证、审稿式反驳、删去证据不足的结论；用户决定投稿与发布。

## 每轮共同交付

实际代码 SHA/dirty 状态；Issue/阶段；改了什么及为何；执行命令和 exit code；文件哈希与依赖；通过/失败项；原始与聚合结果位置；资源；偏离规格；下一步建议。没有执行就写 not_run，不能用预计数字填写结果。
