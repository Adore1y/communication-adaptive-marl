# 项目: 大规模多智能体机器人协同学习系统（示例）

## 快速开始

```bash
# 1. 上传代码到HPC
scp -r marl/examples/project1_multirobot zlei519@hpc2login.hpc.hkust-gz.edu.cn:~/projects/

# 2. 提交训练作业
sbatch train_multirobot.sh

# 3. 查看训练状态
squeue -u zlei519
```

## 文件结构

```
project1_multirobot/
├── README.md
├── train_multirobot.sh          # SLURM作业脚本
├── multirobot_training.py        # 主要训练代码
└── multiagent_env.py             # 多智能体环境
```

## 实验配置

- **智能体数量**: 5 → 20 → 100
- **训练步数**: 10M
- **并行度**: 16个GPU（示例脚本，按需改为单节点8卡）


