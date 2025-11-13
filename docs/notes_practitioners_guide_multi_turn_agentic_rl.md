# A Practitioner’s Guide to Multi-Turn Agentic Reinforcement Learning（阅读笔记）

> 目的：提炼多轮（multi-turn）Agentic RL 的工程实践要点，并映射到本项目（方向二：通信自适应 MARL + Agentic RL）的实现与评测。

## 1. 多轮代理基本循环（Loop Patterns）
- 典型结构：Plan → Act → Observe → Reflect → (Optional Tool Use) → Memory Update。
- 关键超参：反思间隔（steps/episodes）、工具调用预算（每N步允许次数β）、记忆读写策略（召回Top-K vs. 语义检索）。
- 建议：
  - 把“反思”显式建模为策略的一个动作/子目标更新步骤（可学习的触发阈值）。
  - 工具调用预算纳入价值函数（成本化），避免“无边界思考”。

## 2. 记忆与反思（Memory & Reflection）
- 经验：存储“可压缩的教训摘要”（失败原因、重要状态、无效子策略），优先结构化而非纯文本。
- 触发：
  - 基于置信度/进度的门控触发反思（例如目标长期未推进）。
  - 失败轨迹→摘要→偏置后续策略/价值（RLAIF/自监督皆可）。
- 度量：反思有效率（Reflection Success Rate）、反思成本（时间/步数）。

## 3. 奖励与信用分配（Rewards & Credit Assignment）
- 多轮任务稀疏奖励常见，建议：
  - 任务分解（子目标图/Options）+ shaped rewards；
  - 跨回合信用分配（例如对子目标完成前的关键反思赋予信用）；
  - 工具/反思成本入账（避免无效思考/调用）。

## 4. 数据与日志（Data & Logging）
- 强制标准化日志：
  - 轨迹（含反思/工具选择）、记忆读写、置信度、调用预算消耗、失败模式标签。
- 支持离线复盘与可证伪：
  - 提供“固定随机种子+固定地图/任务split”与可回放脚本；
  - 轨迹对齐工具结果（tool I/O）以复现实验。

## 5. 评测协议与指标（Evaluation）
- 成功率（%）、平均Episode长度、子目标完成率、工具调用次数/成功率、反思触发率/有效率、学习曲线AUC。
- 成本维度：交互步数、工具成本（API调用、延迟）、反思时间。
- 鲁棒性：扰动（延迟/丢包/噪声）、长序列记忆毒化、任务分布漂移。

## 6. 安全与约束（Safety/Constraints）
- 门控：对高风险动作/调用设置白名单/黑名单、速率限制、审核器（rule/critic）。
- 约束优化：将预算、时延、安全规则写入约束（Lagrangian/penalty）。

## 7. Scaling 与超参（Scaling & HParams）
- 建议起点：
  - num_envs/GPU=64、n_steps=256、batch ≈ 16k、PPO epochs=2–4、clip=0.2；
  - 反思间隔：5e4–1e5 steps；记忆容量：1e4–5e4；工具预算β=5/1e4 steps。
- 伸缩实验：随预算β、反思间隔、记忆容量的性能-成本曲线。

## 8. 典型失败模式（Failure Modes）
- 过度反思（反思成瘾）、工具幻觉（胡乱调用）、记忆污染（错误经验放大）、子目标循环（卡在局部最优）。
- 缓解：预算+置信门控+一致性检查+回溯剔除机制。

## 9. 融入本项目（方向二）的落地改造
- 训练代码：
  - 在 `agentic_rl/agentic_training.py` 增加：
    - 反思触发间隔/门控、工具预算β与代价项、记忆读写统计日志；
    - 轨迹结构化日志（反思摘要、工具I/O、置信度）。
- 评测脚本：
  - 在 `marl/scripts/eval_comm.py` 的CSV中加入反思/工具列（如 reflection_rate, reflection_success, tool_calls, tool_success）。
  - 伸缩/成本曲线：生成 `成功率-预算β`、`AUC-反思间隔`、`性能-成本(Pareto)` 曲线。
- 实验矩阵：
  - MiniGrid/BabyAI：DoorKey/PutNext（5 seeds，5M steps）；
  - 通信扰动（方向二特有）：带宽/时延/丢包+节点失效下的反思与工具调用稳健性。

## 10. 参考（相关/相近实践）
- Reflexion（ICLR 2024）：https://arxiv.org/abs/2303.11366
- ReAct（NeurIPS 2023）：https://arxiv.org/abs/2210.03629
- Voyager（2023）：https://arxiv.org/abs/2305.16291
- Toolformer（2023）：https://arxiv.org/abs/2302.04761
- NLE（NeurIPS 2020）：https://arxiv.org/abs/2006.13760
- BabyAI/MiniGrid：https://arxiv.org/abs/1810.08272 / https://arxiv.org/abs/1804.00595

> 备注：若后续获取到该指南的正式链接与BibTeX，可在此处补充完整引用条目。


