# 通信自适应的大规模协作 MARL 研究计划

## 1. 研究目标与假设
- 目标：在现实网络约束（带宽/时延/丢包/节点失效）下，提出“分层通信 + 自适应预算”的协作 MARL 框架，在 100+ 智能体规模上取得稳定收益，并给出可复现实验协议与伸缩/鲁棒基准。
- 假设：
  - H1：显式“何时通信/与谁通信/传什么”的预算化决策，比固定通信或启发式稀疏更优（性能-通信 Pareto 前沿更优）。
  - H2：全局—区域中继—个体的分层设计可将复杂度从全连通降至近线性，保持随规模增长的收敛稳定性。
  - H3：在 MAPPO/HAPPO 框架中引入通信门控与信赖域更新，可在约束场景下获得更稳定学习曲线。

## 2. 预期贡献
- C1：自适应预算分层通信：全局-区域中继-个体结构 + 学习式预算与置信触发（when/whom/what）。
- C2：可复现实验协议：伸缩（16→200 体）与鲁棒（BW/Lat/Loss/Fail）基准与脚本。
- C3：系统化消融与前沿曲线：预算门控/中继层/信赖域等模块的贡献与性能-通信 Pareto 前沿。

## 3. 方法概述
- 分层结构：
  - 全局层：低维高频指令（任务状态/群体约束）。
  - 区域中继层：Transformer/GAT 聚合与转发（降低全连通）。
  - 个体层：轻量执行策略（部分可观测）。
- 自适应预算：
  - 学习“何时/与谁/传何种信息”的门控，预算 β 与通信代价纳入优化（惩罚/拉格朗日）。
  - 置信触发：不确定性/信息增益驱动触发阈值。
- 稳定优化：
  - 在 MAPPO/HAPPO/HATRPO 上加入通信门控，结合价值分解（VDN/QMIX）可选模块。

## 4. 实验设计
### 4.1 场景与规模
| 场景 | 智能体规模 | 约束 | Baselines | Ours | 种子 | 步数 |
|---|---|---|---|---|---|---|
| 仓储协同搬运 | 16 → 64 | 无/带宽/时延 | MAPPO/QMIX/HAPPO | 分层+预算 | 5 | 5M |
| SAR（UAV+UGV） | 32 → 128 | 丢包/失效 | HAPPO/HATRPO | 分层+异构注意 | 5 | 8M |
| 动态编队控制 | 64 → 200 | 带宽+时延 | MAPPO+手工通信 | 分层+置信触发 | 5 | 10M |

### 4.2 指标
- 主指标：成功率（%）、完成时间/延迟、通信使用率/预算、鲁棒性（延迟/丢包/失效）。
- 伸缩：性能-规模曲线（N=16/32/64/128/200）。
- 成本：性能-通信 Pareto 前沿。
- 统计：均值±95%CI（5 seeds），t-test/bootstrapping；固定随机种子与任务 split。

### 4.3 消融与敏感性
- 去除模块：预算门控、区域中继层、置信触发、信赖域、价值分解。
- 敏感曲线：性能-预算β、通信-性能前沿、参数热力图。

## 5. 复现与工程
- 目录：`marl/`
  - `docs/`：`direction2_adaptive_marl.md`、本计划
  - `scripts/`：`array_train.slurm`、`eval_comm.py`、`hyperparam_grid.yaml`
  - `experiments/agentic_rl/`：训练样例与结果模板
  - `examples/project1_multirobot/`：示例环境脚本
- 运行：
  - 训练：`sbatch marl/scripts/array_train.slurm`
  - 汇总：`python marl/scripts/eval_comm.py --results <csv>`
  - 超参：按 `hyperparam_grid.yaml` 使用作业数组/并行试验
- 日志：保存配置、随机种子、指标 CSV、曲线（学习/伸缩/鲁棒/前沿）。

## 6. 计算资源与预算（8 × A800 单节点）
- GPU：8×A800；CPU：≥128 核；RAM：≥512GB。
- 并行策略：
  - 作业数组 8 并发（不同种子/超参/任务）。
  - 每任务 1 GPU + 16 CPU；采样 64 并行环境。
- 时间估算：
  - 5M/8M/10M 步：约 24–72 小时/作业；全矩阵 3–5 天（8 并发）。

## 7. 时间线（12 周）
| 周次 | 任务 | 产出 |
|---|---|---|
| W1–2 | 基线复现（MAPPO/HAPPO/QMIX），协议脚本 | 基线曲线、固定 split、评测脚本 |
| W3–4 | 分层通信实现（全局-中继-个体） | 小规模验证与消融 |
| W5–6 | 自适应预算与置信触发、代价集成 | 预算-性能曲线（小规模） |
| W7–8 | 大规模试验（64/128/200），鲁棒性 | 伸缩/鲁棒/前沿曲线初稿 |
| W9 | 超参搜索与稳定性改进 | 最优配置与 ablation 完成 |
| W10 | 结果清洗与制图 | 论文图表定稿版 |
| W11–12 | 写作与内审（ICML） | 初稿→终稿，补充对照 |

## 8. 风险与应对
- 收敛不稳：启用信赖域、增大 rollout/并行度、学习率 warmup。
- 效果不足：强化现实约束协议，丰富消融，扩大规模与统计显著性。
- 时间不足：优先完成“仓储+编队”两场景全矩阵；SAR 作为次优先级。


## 9. 里程碑检查清单（片段）
- [ ] 基线与协议通过（W2）
- [ ] 分层通信可运行（W4）
- [ ] 自适应预算/置信触发收益（W6）
- [ ] 伸缩/鲁棒/前沿三类曲线齐全（W8）
- [ ] 消融与超参完成（W9）
- [ ] 图表与写作冻结（W10–12）

> 配套文件：`marl/docs/direction2_adaptive_marl.md`（方法与实验大纲）；脚本：`marl/scripts/`；结果模板：`marl/experiments/agentic_rl/results/templates/`。

## 10. 本地（MacBook Pro M4）与 HPC 分工

### 10.1 本地可执行（开发/轻量实验/写作）
- MATLAB（原生 arm64）：数据预处理、小规模仿真可视化、原型验证；批处理示例：
  ```bash
  /Applications/MATLAB_R2024a.app/bin/matlab -batch "run('script.m')"
  ```
- Python（Miniforge/conda，MPS 或 CPU）：环境搭建、单元测试、小规模 Minigrid/Agentic RL 快跑、制图。
  - 环境与依赖：
    ```bash
    conda create -n marl python=3.10 -y
    conda activate marl
    pip install gymnasium minigrid stable-baselines3 numpy pandas matplotlib
    pip install torch torchvision torchaudio
    ```
  - 本地小规模训练（示例，缩小并行与步数）：
    ```bash
    cd ~/Desktop/HPC
    conda activate marl
    python marl/experiments/agentic_rl/agentic_training.py \
      --env minigrid_doorkey --num_envs 8 --total_timesteps 200000 \
      --use_memory --use_reflection --use_tools \
      --eval_interval 20000 --log_dir logs_local --save_dir ckpt_local
    ```
- LaTeX（论文写作与编译）：
  ```bash
  brew install tectonic latexmk
  cd ~/Desktop/HPC/papers/icml && make pdf
  ```
- 本地产出物：
  - 预处理脚本与可视化图；小规模对比曲线（验证变更是否有效）；论文图表草图与摘要/引言撰写；配置与依赖说明。

### 10.2 HPC 专属（规模化训练/伸缩与鲁棒评测）
- 大规模并行训练（作业数组/多种子/多任务）；伸缩曲线（N=16→200）、鲁棒扰动（BW/Lat/Loss/Fail）。
- 全面消融与超参搜索；统计显著性（5 seeds）与 Pareto 前沿绘制。
- 产出：最终曲线与 CSV、最优配置、可复现实验日志与脚本。 
