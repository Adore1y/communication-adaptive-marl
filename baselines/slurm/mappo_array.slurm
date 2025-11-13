#!/bin/bash
#SBATCH --job-name=mappo_array
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-7
#SBATCH --time=72:00:00
#SBATCH --output=marl/baselines/slurm/logs/mappo_%A_%a.out
#SBATCH --error=marl/baselines/slurm/logs/mappo_%A_%a.err

module load python/3.9 || true

SEED=${SLURM_ARRAY_TASK_ID}
ENV_NAME=${ENV_NAME:-simple_spread_v3}
STEPS=${STEPS:-5000000}

python marl/baselines/run_mappo_rllib.py \
  --env ${ENV_NAME} --steps ${STEPS} --seed ${SEED} \
  --out marl/baselines/results_mappo.csv --workers 8

