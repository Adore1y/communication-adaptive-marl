#!/bin/bash
#SBATCH --job-name=agentic_array
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --time=72:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

module load python/3.9
module load cuda/11.8

export OMP_NUM_THREADS=16

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs checkpoints

SEED=${SLURM_ARRAY_TASK_ID}

python "$SCRIPT_DIR"/agentic_training.py \
  --env minigrid_doorkey \
  --algo PPO \
  --num_envs 64 \
  --total_timesteps 5000000 \
  --use_memory \
  --use_reflection \
  --use_tools \
  --reflection_interval 50000 \
  --tool_budget_per_10k 5 \
  --eval_interval 50000 \
  --save_dir checkpoints/seed_${SEED} \
  --log_dir logs/seed_${SEED}
