# 交给 Codex：首轮只执行 P0 / P1

你是 idler11/MOE 项目的实现与实验执行者。研究负责人已在仓库给出理论与任务；不要另选方向，也不要先写论文或复述一遍计划就结束。

## 先读取并检查

读取 AGENTS.md 指定的全部文件，检查 git status、当前 branch、remote。保留用户改动。在独立工作分支 codex/p0-p1-math-core 上工作；分支已存在则检查后继续，不重建覆盖。

用几句话确认你理解：

- 我们优化实际可部署的量化码，不是逐 token 自由选专家。
- Delta L = Delta L_fixed + J_event；J_event 在候选终点、新输入上计算且有符号。
- finite empirical gradient 不必错误；困难是有限步候选跨分支。
- 现在没有预训练模型收益，不能把 legacy toy 数字当作论文结果。

## P0：实际执行

1. 记录真实 Python、NumPy、SciPy、PyTorch、pytest、操作系统与设备可用性。不要重装 CUDA 或替换工作的 PyTorch。
2. 执行 python scripts/validate_handoff.py。
3. 在已有依赖可用的项目环境中执行 python scripts/recheck_legacy.py --output artifacts/p0_recheck_001；该目录若存在，换新 run_id，不覆盖。
4. 检查哈希、退出码、数值差异、stdout/stderr。任何缺依赖、超时或不匹配都要报告，不伪造通过。
5. 写 reports/P0_environment.md，注明本机证据与仓库历史证据的区别。不要提交包含账户路径/凭据的大日志。

## P1：请写代码和测试，不只是给建议

按 docs/04_ENGINEERING.md 实现 src/moe_event 的最小核心：toy MoE、路由几何、合法固定尺度候选、有限步分解、候选坐标导数、评分器和选择遗憾指标。先实现直观 dense reference，再实现 event-only 路径供等价性检查。不要接大模型。

按文档逐项实现 pytest 测试，覆盖两种 gate 模式、无事件/多专家替换/近 tie、负事件项、零空间非功能不变、full-Hessian/Gauss-Newton 区分、候选合法性/no-op、shared-weight reachability、dense/event-only 一致性。float64 用于数学参考，float32 用于数值误差检查；边界稳定与不稳定情况分开报告。

已有 legacy 代码是对照，不要在原文件上重构。新测试应具有独立实现或手工反例，避免两个函数共享同一个 bug。

新增可运行入口 scripts/run_p1_checks.py，使用 configs/p1_smoke.json；运行新测试和固定种子 smoke。P1 只检查正确性，不要求新方法获胜；不要自动跑 P2 的统计 sweep。

写 reports/P1_math.md，使用 reports/REPORT_TEMPLATE.md 的证据格式；记录未支持的模型假设、失败与实际命令。更新 PROJECT_STATE.json 的实现/测试状态，但保留下一阶段未获研究批准。

## 完成后的提交

在当前工作分支提交代码、小型结果与报告；有远程权限则 push 并开供审查的 PR，不自行合并。没有权限则保留本地 commit 并给出 SHA 和 diff，不虚构已推送。

最终回复提供：分支/commit/PR、命令与退出码、测试数与失败项、结果文件、数值差异、下一步障碍。结束本轮，不下载模型、不训练、不开始 P2。之后用户会把结果交给 ChatGPT 做理论与实验审查。
