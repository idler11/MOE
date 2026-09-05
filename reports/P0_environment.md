# P0｜环境与隔离 legacy 审查包复现

状态：needs_review（隔离运行完成，但有一项历史数值在预设容差外）。

## 任务与证据定位

Issue：START_PROMPT.md 的 P0。

工作分支 / code commit / dirty 状态：`codex/p0-p1-math-core` / `9cb4dce2bd5da81a65e795a7181e632c2e44edf1` / clean at legacy-run start。

run_id：`p0_recheck_001`。

配置路径与 SHA-256：none（wrapper 使用其默认 `rtol=1e-6`、`atol=1e-9`、`timeout=180s`）。

生成器/数据 split/hash：不适用；这是历史随机 synthetic audit 的隔离复现，不是本轮数据集或 holdout 评测。

模型与 revision：none。

本机环境：Windows `10.0.26200`、CPython `3.9.10`、NumPy `2.0.2`、SciPy `1.13.1`、PyTorch `2.7.1+cpu`；`torch.cuda.is_available()` 为 false、设备数为 0、CUDA version 为 null。初始解释器没有 pytest；P1 随后在仓库 `.venv` 中单独安装 pytest，未替换 PyTorch 或 CUDA。

历史 evidence 环境是 `legacy/moe_theory_audit/environment.json` 中的 Linux / CPython `3.13.5` / NumPy `2.3.5` / SciPy `1.17.0` / PyTorch `2.10.0+cpu`。两者不得混写：后者是仓库历史元数据，前者才是本机实际运行证据。

## 实际执行

| 命令 | 退出码 | 结果 |
|---|---:|---|
| `python scripts/validate_handoff.py` | 0 | 22 个 required files 存在，4 个 immutable legacy 文件哈希通过，无错误。 |
| `python scripts/recheck_legacy.py --output artifacts/p0_recheck_001` | 1 | wrapper 标记 `needs_review`；隔离 legacy 子进程自身退出码为 0，无 timeout。 |

预检版本探测的首次简化命令因 pytest 缺失退出 1；一次兼容性探测因 Python 3.9 的 `importlib.metadata` 没有所查询 API 退出 1。两者均是环境探测命令的失败，不是仓库代码或 legacy 脚本失败；随后使用直接模块导入的容错探测，退出 0 并得到上列版本。未安装/升级任何 GPU 包。

隔离 wrapper 的 stdout/stderr 位于：

- `artifacts/p0_recheck_001/stdout.txt`（legacy synthetic audit 输出）
- `artifacts/p0_recheck_001/stderr.txt`（空）
- `artifacts/p0_recheck_001/environment.json`
- `artifacts/p0_recheck_001/summary.json`

这些 artifacts 被 `.gitignore` 排除；本报告仅记录小型、可审查摘要。

## 结果

`validate_handoff.py` 确认 manifest 中 4 个原始文件的 SHA-256 正确。`recheck_legacy.py` 在新的 `artifacts/p0_recheck_001/source/` 副本中运行，开始和结束时的源哈希均一致；原始 `legacy/moe_theory_audit/` 未被写入。

隔离子进程耗时 `9.4906226s`，未超时。结果 JSON 可解析；除下项外逐字段比较通过：

| 字段 | 历史值 | 本机隔离值 | 绝对差 |
|---|---:|---:|---:|
| `routing_geometry.nullspace_max_gate_probability_change` | 0.11631804810019553 | 0.11783651643956983 | 0.00151846833937430 |

该输出来自 legacy 脚本中 `np.linalg.svd` 得到的非唯一 nullspace basis，再用 RNG 坐标乘该 basis。不同 NumPy/线性代数实现可能给等价 nullspace 的不同基坐标，从而改变这个汇总值；这是对环境差异的合理解释，尚不是已验证的根因。因此没有放宽容差、修改历史结果、重跑覆盖或把本机复现写成通过。

本次结果文件的 SHA-256 为 `bc94d7bcf93250756bc5c4ff14b2e707d930423cc8fe3c48f53ac81e7b066f17`；其字节哈希不同于历史结果不是判定标准，逐字段容差比较才是。

## 正确性与反例

immutable legacy 哈希：通过（4/4，运行前后相同）。

数值比较：1 个差异，预设 `rtol=1e-6` / `atol=1e-9` 下不通过。没有将此差异删掉或用新结果覆盖历史 `results.json`。

本 P0 复现不使用候选选择、校准、holdout 或 oracle；不存在选择泄漏结论。

## 资源与失败

设备：CPU；GPU 未使用。实际 elapsed：`9.4906226s`。峰值显存、host RAM：null（未测）。

失败项：上述单一 legacy 数值差异；pytest 在初始解释器缺失（P1 已以项目本地虚拟环境解决）。没有 timeout、OOM、模型下载、训练、CUDA 重装或 PyTorch 替换。

## 与规格的偏离

none。隔离目录不存在时按指定 run_id 创建；不修改 legacy 原件，不调整容差以获取通过。

## 研究结论与下一步

本机已执行完整性校验与隔离 legacy 复现，确认原始证据未变，但不能声称本机在当前容差下完全复现历史数值。该 audit 仍只是 CPU random toy 证据，不支持预训练模型、泛化、速度或研究有效性主张。

需要研究审查决定：是否接受 SVD-basis/environment 差异解释、是否固定历史环境做额外诊断，或将此项保留为 P0 阻塞。P2 及后续阶段未获批准，未执行。
