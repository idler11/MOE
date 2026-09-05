# 04｜Codex 工程接口与验收测试

## 已有与待实现

已有：交接文档、legacy 审查包、stdlib 哈希检查与隔离复现工具。
待 Codex 实现：src/moe_event、tests、scripts/run_p1_checks.py 及以后所有科学实验入口。不要假装这些已存在。

优先 Python + NumPy + PyTorch，pytest 测试。先检查已装版本；不绑定猜测中的 CUDA/torch 组合。允许项目专用环境，不运行全局升级。P0/P1 CPU，float64 reference。之后记录实际版本形成可复现 lock，不在未验证前随便写版本号。

## 建议模块及接口契约

| 模块 | 职责与契约 |
|---|---|
| types.py | QuantizationSpec、CandidateSet、RouteState、LossDecomposition、ScoreResult、RunManifest；shape/dtype/units 明确 |
| toy.py | deterministic RMSNorm + bias-free router + SiLU experts + residual；forward(h, forced_sets=None, normalize_topk=bool)；forced 仅固定集合 |
| routing.py | stable top-k、集合 canonicalization、centered router、endpoint events、first_exit_time；不支持架构显式失败 |
| candidates.py | 固定尺度有限码、合法 floor/ceil 修改、clipping/dedup/no-op、shared Q(z)；不读取目标损失挑坐标 |
| decomposition.py | actual_delta/fixed_delta/signed_event，逐 token 与 mean；new-input semantics |
| derivatives.py | p 维 g/H，HVP、可选 GN 独立命名；固定旧 route indices 但不 detach gate 权重 |
| scoring.py | first/second/event-corrected/nonnegative ablations；无 holdout/oracle 参数 |
| metrics.py | opportunity、regret、holdout diagnostic、MAE、rank、undefined/null 处理 |
| audit.py | allowlist 环境、命令/配置/seed/hash/exitcode/cost；拒绝覆盖 run_id |

张量实现可使用行向量 [N,d]，但必须记录与文档列向量的转置：hidden=b+a@Q.T，raw_logits=hidden@B.T。不要出现可广播但错误的 gamma、M 或 token 维度。

reference dense evaluator 可算全部专家用于 oracle；event-only evaluator 对事件 token 的旧/新专家集合并集执行。全专家 gate logits/softmax 仍按真实规则算。hook 或计数器应能证明 expert 实际未被调用，而不是先 dense 再 mask。

候选表以 candidate_id、码修改列表/哈希标识；排序方法只能读 calibration 评分。真实 oracle 表由 evaluator 独立保存，不能由 scorer 引用。

## P1 必需测试矩阵（逐项写实际通过状态）

T01 finite-vs-population scalar counterexample：有限 FD=0、总体解析导数=0.5，不能混为一谈。
T02 RMSNorm/bias-free raw logits 与 normalized logits 的 Top-k 在严格无 tie 情形一致；两种 gate 模式。
T03 first-exit 与沿线精确候选路由一致，覆盖无退出、终点边界、多竞争专家；不能只查 k/k+1。
T04 centered nullspace 保持集合但存在 gate/功能变化的明确反例。
T05 finite-step 逐 token 和 mean 恒等式，zero-step/no-event/multi-expert replacements 均覆盖。
T06 signed term=2 residual^T M jump+jump^T M jump；正负实例都有。
T07 用旧输入估 jump 或强行丢交叉项会失败的反例（负测试）。
T08 forced route recomputes gate weights；全 E softmax 不得错误地变成 top-k renorm。
T09 所有码合法、fixed bit/group/scale、no-op 存在、组合后无越界、duplicates 记录；same z 同时作用全部 token。
T10 shared-weight reachability 标量反例；禁止逐 token oracle 冒充候选。
T11 固定分支 grad/H 与稳定分支 finite difference/HVP 一致；非零 residual 下完整 H 与 GN 差异单列。
T12 event-corrected score 的误差随平滑余项变化；没有宣称未知 beta 已数值认证。
T13 dense/event-only outputs/events/grad（适用部分）一致，两个 gate 模式都测；expert call counter 确实减少而不声称速度。
T14 stable tie policy、near-tie flags、empty event set、k=1 与 k=E、非法 k/bias 模式 rejection。
T15 metrics 包含 zero opportunity、constant rank、负改进、两个方法选同一候选等合法情况；JSON 不含 NaN/Infinity。
T16 calibration/holdout 独立；选中码从 calibration 保存后，在 holdout 不能再次 argmin；加入故意 leakage 的负测试。
T17 legacy 哈希不变；重复 run_id 拒绝覆盖；输出的 manifest 能定位输入和代码。

float64 初始 allclose 可取 atol=1e-10, rtol=1e-8，适用于小型已缩放 toy；finite difference 应扫 step 稳定区间而非一个神奇步长。float32 容差按 reference 误差测量预定。两种精度的收益不是单元测试断言。

## 4090 后续资源策略（尚未实测）

先检查真实 VRAM/host RAM/disk，不把激活参数量当总权重显存。分层缓存输入，teacher 与量化候选顺序跑；只对小 p 坐标求导，冻结其余权重，仍允许梯度穿过必要的冻结运算。不要误用 no_grad 阻断待求导路径。

p=8 只说明 Hessian 有 64 个元素，不说明构造便宜；实际前后向计入账本。长文本、双模型常驻、全部候选激活同时保留都禁止作为默认实现。离线缓存仅存本项目生成数据，不入 Git。

没有写研究核心之前不添加 Web UI、服务、数据库、任务队列、复杂 distributed runtime 或定制 CUDA kernel。
