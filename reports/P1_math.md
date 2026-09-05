# P1｜独立数学核心、测试与固定种子 smoke

状态：executed，needs_review（实现与测试完成；研究审查尚未批准后续阶段）。

## 任务与证据定位

Issue：START_PROMPT.md 的 P1。

工作分支 / code state：`codex/p0-p1-math-core` / `01beec73354f189c04b799ca33a856b043a63bc8` / `dirty=false`。完整 pytest 与 smoke 均在这个 clean source commit 上执行；后续 evidence-only commit 不改 P1 源码。

run_id：`p1_smoke_002`。

配置路径与 SHA-256：`configs/p1_smoke.json` / `1faef9de758233de722bb6d32c277fb52e5f86827cf0a759aa0c6b1c5cdb51de`。

生成器/数据 split/hash：脚本以 seed `0,1,2` 在 CPU 生成独立 toy inputs、bias、teacher Q 与 4-coordinate 固定尺度候选；没有外部数据，没有 holdout 用于选择。P1 不作 P2 的 calibration/holdout 泛化结论。

模型与 revision：none；`ToyMoE` 是新写的 deterministic RMSNorm + 无 bias router + SiLU expert reference，不是预训练模型或真实架构适配器。

环境版本、dtype、设备：Windows `10.0.26200`，CPython `3.9.10`，NumPy `2.0.2`，SciPy `1.13.1`，PyTorch `2.7.1+cpu`。参考为 CPU float64，二级数值检查为 float32。为运行 pytest，在仓库 `.venv`（`--system-site-packages`）中单独安装 pytest `8.4.2`；未升级 pip、未替换 PyTorch、未使用 CUDA/GPU。

## 实际执行

| 命令 | 退出码 | 结果 |
|---|---:|---|
| `python -m compileall -q src scripts tests` | 0 | 新 Python 源和测试语法通过。 |
| `.\\.venv\\Scripts\\python.exe -m pytest -q` | 0 | `22 passed in 4.58s`，在 clean source commit 上执行。 |
| `.\\.venv\\Scripts\\python.exe scripts/run_p1_checks.py --config configs/p1_smoke.json --output artifacts/p1_smoke_002` | 0 | 6 个固定 seed×gate smoke case；manifest 记录 `01beec7...` / `dirty=false`。 |

pytest 初次不可用是本机解释器缺少包，而不是用例跳过；在项目本地环境安装后运行。没有跳过的 P1 测试。

原始 smoke JSON 位于 `artifacts/p1_smoke_002/summary.json`（ignored）；已提交的小型摘要位于 `reports/p1_smoke_summary.json`。

## 结果

新增模块：

- `src/moe_event/toy.py`：完整 dense reference、all-token route-only path 与 event-token 去重 union evaluator；重合的 old/new expert 每 token 仅运行一次，forced route 只固定 indices，在候选输入上重算 logits/softmax/gate。
- `src/moe_event/routing.py`：stable descending Top-k（平分时较小 expert id 优先）、canonical set、centered router、first exit、near-tie/event 检测；bias router 显式不支持。
- `src/moe_event/candidates.py`：固定 scale、合法 floor/ceil、midpoint/stable-index 坐标、no-op、组合合法性与 duplicate/rejected-mask 记录；候选生成不接收 target/loss。
- `src/moe_event/decomposition.py`：dense 的完整逐 token `Delta L = Delta L_fixed + J_event`，以及仅精确计算 `J_event` 的 event-only 路径；q0 route cache 可跨候选复用，非事件 token 不执行 experts，也不伪造其 total loss。
- `src/moe_event/derivatives.py`：固定旧 route indices、连续候选坐标的完整 grad/Hessian/HVP，另行命名 Gauss--Newton，不把两者混同。
- `src/moe_event/scoring.py` 与 `metrics.py`：calibration-only 选择、post-selection oracle/holdout diagnostic、zero-opportunity/null rank/JSON-finite 处理。
- `src/moe_event/audit.py`：legacy hash 验证与不可覆盖 run directory；`scripts/run_p1_checks.py` 为配置化 smoke 入口。

Smoke 结果（6 case 聚合）：

| 项目 | 实测值 |
|---|---:|
| 最大 `actual - fixed - event` | `3.469446951953614e-18` |
| 最大 signed event 公式误差 | `3.0531133177191805e-16` |
| 最大 dense/event-only aggregate signed event 差 | `1.0408340855860843e-17` |
| 最大 dense/event-only event 差 | `3.3306690738754696e-16` |
| float32 最大 output 差 | `8.883074045229478e-07` |
| float32 / float64 route sets 一致 | 6/6 |
| dense expert calls / case | 576 |
| event-only expert calls / case | 18–21 |
| event token 数 / case | 6–7 |
| `max|H - GN|` / case | `2.683e-4`–`1.980e-3` |

seed 2 的两个 gate mode 都出现负的 signed event（分别约 `-0.0078054`、`-0.0207986`），因此测试/结果不假定 event 非负。event-only 对全部 token 仅计算 router；只对 6–7 个 event token 运行 q1 上 `S_old ∪ S_new` 的去重 expert union，q0 route cache 在每个 case 中复用。expert call 计数不是墙钟速度或部署效率主张。

## 正确性与反例

| 工程规范项 | 覆盖与状态 |
|---|---|
| T01 | 独立 finite-vs-population scalar counterexample；通过。 |
| T02 | RMSNorm ordering 与两种 gate mode；通过。 |
| T03 | all-pair first exit、无退出、endpoint boundary、多竞争 expert；通过。 |
| T04 | centered nullspace 保 route set 但改变 gate/function；通过。 |
| T05 | zero step、no event、多 expert 替换的逐 token/mean identity；通过。 |
| T06–T08 | 正负 signed term、旧输入/丢交叉项反例、forced gate 重算；通过。 |
| T09–T10 | fixed-scale legal/no-op/duplicate、bits/codebook 与 zero-point 非法反例、shared-weight reachability；通过。 |
| T11–T12 | 五个数量级的 finite-difference 稳定区间、HVP、完整 H≠GN、事件后剩余 smooth Taylor error；通过。 |
| T13–T14 | dense 与 true event-only `J_event` parity、q0 route cache 复用、event union 去重/no-event 零 expert call、tie/k/bias rejection；通过。 |
| T15–T16 | zero opportunity、constant rank、负改进、同选项、strict JSON、calibration/holdout 隔离与故意 leakage 负测试；通过。 |
| T17 | legacy 4-file hash、run-id overwrite refusal；通过。 |

选择函数 `select_from_calibration` 没有 holdout/oracle 参数；故意传入 `holdout_losses` 会抛出 `TypeError`。测试中 oracle table 只在 `metrics.py` 的选择后诊断中出现。没有把候选的实际 loss 或研究方法的收益作为测试通过条件。

## 资源与失败

实际 smoke elapsed：`15.6399475s`；CPU only。峰值显存/host RAM：null（未测）。

P1 测试失败项：无（22 passed）。P1 smoke failure list：空。P0 legacy numeric mismatch 单列在 `reports/P0_environment.md`，没有被 P1 覆盖或解释为通过。

未支持/明确边界：带 bias router、grouped top-k、capacity drop、stochastic/batch-dependent routing、非线性上游、多层相加、真实模型 adapter、packed low-bit kernel/per-group deployment format；其中 P1 toy 对 bias 显式拒绝。没有下载模型、训练、跑 P2 sweep 或声称质量/速度/新颖性收益。

## 与规格的偏离

none。P1 只用了 self-generated CPU toy；没有读取 holdout 或 oracle 以作候选选择，未改变 legacy 原件，未执行 P2。

## 研究结论与下一步

本轮支持的是 P1 reference/core 的数学接口、反例和 dense/event-only 一致性；不支持事件校正优于基线、真实 MoE 有收益、泛化、加速或创新性结论。P0 的历史数值差异仍需审查。

请由研究负责人审查 P0 mismatch、P1 接口/测试与共享候选语义；在明确批准前，P2 仍关闭且不会自动执行。
