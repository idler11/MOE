# P?_报告标题

状态：not_run / blocked / executed / needs_review（如实选择）。

## 任务与证据定位

Issue：
工作分支 / code commit / dirty 状态：
run_id：
配置路径与 SHA-256：
生成器/数据 split/hash：
模型与 revision（没有使用则写 none）：
环境版本、dtype、设备：

## 实际执行

命令：
退出码：
开始/结束时间：
stdout/stderr 位置（公开前脱敏）：
测试数、通过/失败/跳过与原因：

## 结果

填写真实小型 JSON/CSV 路径与必要聚合。未运行字段填 null，不使用预计数字。说明 loss 归一化、oracle 使用范围、holdout 是否参与选择、同算力比较是否完成。

## 正确性与反例

恒等式最大误差、官方/自定义 parity、negative tests、near-tie、unsupported 情况。

## 资源与失败

实际耗时、峰值显存/host RAM（未测填 null）、extra expert calls、导数与筛选开销。
零/负收益、误差、OOM、timeout、dependency 问题均保留。

## 与规格的偏离

修改了什么、为什么、是否获得授权。没有偏离写 none。

## 研究结论与下一步

仅本阶段证据支持什么、不支持什么；需要 ChatGPT 审查的问题。下一阶段未批准前不执行。
