# Agentic RL（记忆-反思-工具）

## 快速开始
```bash
scp -r agentic_rl zlei519@hpc2login.hpc.hkust-gz.edu.cn:~/projects/
sbatch agentic_rl/train_agentic_rl.sh
```

## 结构
```
agentic_rl/
├── README.md
├── train_agentic_rl.sh           # SLURM 作业数组（8并发）
├── agentic_training.py           # 训练入口（含记忆/反思/工具钩子）
└── results/
    ├── templates/
    │   ├── results_template.csv
    │   └── ablation_template.csv
    └── eval/
        └── compare_baselines.py
```
