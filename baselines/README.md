# MARL 基线（RLlib）

## 依赖
```bash
conda create -n marl-baselines python=3.10 -y
conda activate marl-baselines
pip install -r marl/baselines/requirements.txt
```

## 本地快速运行（小步数Sanity）
```bash
# QMIX / MADDPG / PPO / MAPPO(RLlib-PPO)
python marl/baselines/run_qmix_rllib.py   --env simple_spread_v3 --steps 200000 --seed 0 --out marl/baselines/results_qmix.csv   --curve_out marl/baselines/curves_qmix_seed0.csv
python marl/baselines/run_maddpg_rllib.py --env simple_spread_v3 --steps 200000 --seed 0 --out marl/baselines/results_maddpg.csv --curve_out marl/baselines/curves_maddpg_seed0.csv
python marl/baselines/run_ppo_rllib.py    --env simple_spread_v3 --steps 200000 --seed 0 --out marl/baselines/results_ppo.csv    --curve_out marl/baselines/curves_ppo_seed0.csv
python marl/baselines/run_mappo_rllib.py  --env simple_spread_v3 --steps 200000 --seed 0 --out marl/baselines/results_mappo.csv  --curve_out marl/baselines/curves_mappo_seed0.csv
```

输出汇总CSV列：`scenario,algo,seed,total_steps,episode_reward_mean,notes`
曲线CSV列：`scenario,algo,seed,step,reward`

## 合并与作图
```bash
# 基线最终对比（柱状）
python marl/baselines/merge_and_plot.py \
  --inputs marl/baselines/results_qmix.csv marl/baselines/results_maddpg.csv marl/baselines/results_ppo.csv marl/baselines/results_mappo.csv \
  --out_csv marl/baselines/results_all.csv --out_fig marl/baselines/results_bar.png

# 学习曲线（均值±95%CI），自动合并 curves_*.csv
python marl/baselines/merge_curves_and_plot.py \
  --inputs marl/baselines/curves_*.csv \
  --out_csv marl/baselines/curves_merged.csv --out_fig marl/baselines/curves_ci.png
```

## HPC 作业数组（8 seeds 示例）
```bash
sbatch marl/baselines/slurm/qmix_array.slurm
sbatch marl/baselines/slurm/maddpg_array.slurm
sbatch marl/baselines/slurm/ppo_array.slurm
sbatch marl/baselines/slurm/mappo_array.slurm
```

## 结果汇总
```bash
python marl/scripts/eval_comm.py --results marl/baselines/results_qmix.csv
```
