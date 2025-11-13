# MARL 专用目录（通信自适应与大规模协作）

## 结构
```
marl/
├── README.md
├── pyproject.toml                          # Python 包配置
├── .gitignore
├── docs/
│   ├── direction2_adaptive_marl.md
│   └── Research_Plan.md
├── scripts/
│   ├── array_train.slurm
│   ├── eval_comm.py
│   └── hyperparam_grid.yaml
├── src/marl/                               # 框架源码骨架
│   ├── __init__.py
│   ├── config.py
│   ├── communication/
│   │   └── hierarchical.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── registry.py
│   ├── policies/
│   │   ├── __init__.py
│   │   └── policy.py
│   ├── training/
│   │   └── pipeline.py
│   └── utils/
│       └── logging.py
├── tests/
│   └── test_pipeline.py
├── experiments/
│   └── agentic_rl/
│       ├── README.md
│       ├── train_agentic_rl.sh
│       ├── agentic_training.py
│       └── results/
│           ├── templates/
│           │   ├── results_template.csv
│           │   └── ablation_template.csv
│           └── eval/
│               └── compare_baselines.py
└── examples/
    └── project1_multirobot/
        ├── README.md
        ├── train_multirobot.sh
        ├── multirobot_training.py
        └── multiagent_env.py
```

## 使用
- 作业数组训练：`sbatch marl/scripts/array_train.slurm`
- 指标汇总：`python marl/scripts/eval_comm.py --results <csv>`
- 超参搜索：参考 `marl/scripts/hyperparam_grid.yaml`

## 本地开发快速开始

```bash
cd marl
python -m venv .venv                # 或使用 conda/miniforge
source .venv/bin/activate
pip install -e ".[dev]"             # 安装依赖与 pytest
pytest                              # 运行最小单元测试
```

运行占位训练流程：

```bash
python -m marl.training.pipeline
```

（默认会尝试创建 MiniGrid 环境；如未安装 gymnasium-minigrid，则请先 `pip install gymnasium-minigrid`。）

