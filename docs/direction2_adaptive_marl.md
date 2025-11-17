# 通信自适应的大规模协作多智能体强化学习

## 1. 研究概述
- **研究主题**：针对 100+ 智能体协作任务，设计具备“何时通信、与谁通信、传递何种信息”自适应能力的分层 MARL 框架。
- **灵感来源**：价值分解、注意力通信、分层决策方面的研究成果。


## 2. 技术路线
1. **分层结构**：全局协调层（高频低维指令）、区域中继层（Transformer/Graph Attention）、个体执行层（轻量策略网络）。
2. **通信自适应机制**：学习式预算约束（软/硬预算）、信息价值估计、置信触发（confidence gating）。
3. **稳定训练**：参考 MAPPO/HAPPO/HATRPO，设计信赖域与价值分解相结合的更新策略；支持异构智能体。
4. **可扩展实现**：实现 256–512 并行环境采样 + 异步/同步训练。

## 3. 实验矩阵 

| 场景 | 智能体规模 | 描述 | Baseline | Ours | 重复 | 训练步数 |
| --- | --- | --- | --- | --- | --- | --- |
| 仓储协同搬运 | 16 → 64 | 多机器人搬运/避碰 | MAPPO / QMIX | 分层通信 + 自适应预算 | 5 | 5M |
| SAR（无人机+地面车） | 32 → 128 | 搜索救援（异构） | HAPPO/HATRPO | 分层通信 + 异构注意力 | 5 | 8M |
| 动态编队控制 | 64 → 200 | 形状保持 + 避障 | MAPPO + 手工通信 | 分层通信 + 置信触发 | 5 | 10M |
| 对抗协作 (可选) | 24 vs 24 | 混合协作/对抗 | MADDPG / MAAC | 分层通信 + 双重价值分解 | 3 | 6M |

> 建议逐级扩展智能体数量，并在每个规模下复现实验。

## 4. 超参数网格

### 通信预算与门控
- 通信预算 `B ∈ {0.2, 0.4, 0.6}`（表示允许通信的智能体比例）。
- 触发阈值 `τ ∈ {0.3, 0.5, 0.7}`（由置信度/信息增益决定）。
- 区域划分数量 `R ∈ {4, 8}`（区域中继层节点）。

### 训练超参
- PPO/MAPPO：`lr_actor ∈ {1e-4, 3e-4}`，`lr_critic ∈ {1e-4, 3e-4}`，`clip ∈ {0.1, 0.2}`，`λ_GAE ∈ {0.9, 0.95}`。
- HAPPO/HATRPO：trust region 半径 `δ ∈ {0.01, 0.05}`。
- Batch size `∈ {32768, 65536}`（多环境并行）。
- Rollout length `T ∈ {256, 512}`。
- Grad clip `1.0`，value loss coefficient `0.5`。

### 扩展选项
- 价值分解：测试 VDN、QMIX、QTRAN 与自研模型对比。
- 注意力深度：Transformer 层数 `{2, 4}`；头数 `{4, 8}`。

## 5. 评测指标与协议
1. **核心指标**：任务完成率、平均完成时间/延迟、通信带宽使用率、通信成功率、鲁棒性（节点失效/消息噪声）、复杂度伸缩性（性能随智能体数变化曲线）。
2. **工具性指标**：平均奖励、AUC、learning stability（标准差）。
3. **消融**：固定通信预算、去掉区域中继层、禁用置信触发、仅单层策略等。
4. **鲁棒性实验**：
   - 通信延迟/丢包：p = {0.0, 0.1, 0.3}。
   - 节点失效：随机移除 5%/10% 智能体。
5. **统计检验**：对比最强基线（如 MAPPO/HAPPO）做 t-test 或 bootstrap (95% CI)。

## 6. HPC 执行方案
- **硬件**：单节点 8 × NVIDIA A800-SXM4-80GB，CPU ≥128 核。
- **环境采样**：CPU 负责 256–512 并行环境；GPU 执行策略更新（可按任务/种子分配 1 GPU）。
- **调度策略**：
  - 作业数组运行 8 个并行实验（不同种子/超参）——充分利用 8 卡。
  - 如需单实验多卡，可使用多 GPU 的同步梯度聚合。
- **时间估算**：5–10M 步约 48–96 小时；8 并行可在 3–4 天完成一套矩阵。

### 推荐 SLURM 脚本（作业数组）

```bash
#!/bin/bash
#SBATCH --job-name=dir2_array
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-7
#SBATCH --time=96:00:00
#SBATCH --output=directions/scripts/logs/dir2_%A_%a.out

module load python/3.9
module load cuda/11.8
export OMP_NUM_THREADS=16

SEED=${SLURM_ARRAY_TASK_ID}
python agentic_rl/agentic_training.py \
  --env warehouse_multiagent \
  --algo mappo_adaptive \
  --num_envs 64 \
  --total_timesteps 8000000 \
  --use_memory --use_reflection --use_tools \
  --save_dir runs/dir2/seed_${SEED} \
  --log_dir logs/dir2/seed_${SEED} \
  --seed ${SEED}
```

> 若需单实验多卡，可改为 `torchrun --nproc_per_node=8` 并在训练代码中实现同步更新。



## 7. 参考文献（部分）
- Yu et al., *The Surprising Effectiveness of PPO in Cooperative MARL*, 2021.
- Wang et al., *HAPPO/HATRPO*, NeurIPS 2021.
- Lowe et al., *Multi-Agent Actor-Critic*, NIPS 2017.
- Rashid et al., *QMIX*, AAAI 2018.
- Foerster et al., *Counterfactual Multi-Agent Policy Gradients*, AAAI 2018.
- Espeholt et al., *IMPALA*, ICML 2018.
- Mahajan et al., *MAVEN*, NeurIPS 2019.
- Recent adaptive communication works: e.g., *TarMAC*, *ATOC*, *GAT-based MARL*。

---

有关脚本与自动化工具，请查阅 `marl/scripts/` 中的模板，并按需扩展采样/训练流水线。
