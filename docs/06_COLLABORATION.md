# 06｜ChatGPT × Codex × 用户的协作闭环

## 1. 分工不是重复做同一件事

ChatGPT 维护理论、研究目标、文献差异、实验关卡；Codex 负责将规格变成可运行代码、测试与真实报告。用户在本地启动执行并控制资源。GitHub 是三方共享的事实来源，而不是依靠聊天记忆同步。

这里不会自动远程启动用户机器上的 Codex，也不会后台监控 GPU。用户运行 Codex、push 结果并把分支/PR/commit 发到当前会话后，ChatGPT 再通过 GitHub 读取代码与报告审查。没有实际读取就不宣布通过。

## 2. 用户的首轮操作

在合适父目录 clone idler11/MOE，进入仓库，启动本地已经安装可用的 Codex。已有 clone 时先 git status，干净后再 git pull --ff-only；不要覆盖本地工作。

交给 Codex 的一句指令：

> 读取 START_PROMPT.md 和 AGENTS.md，按仓库规格实际完成 P0/P1；在工作分支提交测试与 reports，完成后停下，不进入 P2。

不要指定未经确认可用的模型名或自动升级 CLI。

## 3. Codex 每轮交付

使用 reports/REPORT_TEMPLATE.md。提供实际命令/exit code、测试、结果文件、小型 summary、失败/偏离、资源、Git SHA、可读取的 PR。每条结果可追到 run_id/config/data hash。无运行则 not_run，阻塞则说明实际报错。

本地 artifacts 通常不在 Git；因此报告必须提交安全的聚合结果和最小复现配置。ChatGPT 看不到用户磁盘的 artifact 路径，不能只写“看我本地文件”。大结果需要用户另附或可访问的受控存储，不得把私人日志公开。

## 4. ChatGPT 的审查顺序

读取 PROJECT_STATE 与本轮报告；检查 diff 和数学接口；核对 tests 有没有 oracle 泄漏/共享 bug；检查实际结果和计费/资源；比较强基线；给出：通过并批准下一阶段、需要修正、证据不足、停止/重新定义问题。

通过意味着本阶段证据满足标准，不是论文保证。更新 DECISIONS 并发出下一轮 Issue/明确 prompt。Codex 不凭上一阶段跑完就继续消耗更多资源。

## 5. Issue / PR 规则

阶段 Issue 作为工作单，PR 作为实现审查单。首次 P0/P1 可在同一个分支，但分别提交环境证据与数学核心。每个 PR 包含范围、不包含什么、命令、测试、数据来源、风险、下一步。

不得 force push 或自动合并自己的研究 PR。不要在 unrelated 仓库新增文件。报告误差或负结果不会被视为执行失败，隐藏误差才是失败。

## 6. 后续提示词模板

> 仓库 idler11/MOE；从指定 commit/PR 读取最新 PROJECT_STATE、DECISIONS 和 reports。仅执行 Issue 指定阶段，其他阶段仍关闭。先处理审查意见，保持数据与候选隔离，不改历史证据。实际运行后提交小型可审查结果、命令/退出码与失败项，不自行宣布论文创新性成立。
