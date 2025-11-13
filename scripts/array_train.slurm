#!/bin/bash
#SBATCH --job-name=dir2_array
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-7
#SBATCH --time=96:00:00
#SBATCH --partition=normal
#SBATCH --output=marl/scripts/logs/dir2_%A_%a.out
#SBATCH --error=marl/scripts/logs/dir2_%A_%a.err

module load python/3.9
module load cuda/11.8
export OMP_NUM_THREADS=16

SEED=${SLURM_ARRAY_TASK_ID}
ENV_CFG=configs/dir2/warehouse.yaml
python agentic_rl/agentic_training.py \
  --config ${ENV_CFG} \
  --total_timesteps 8000000 \
  --save_dir runs/direction2/seed_${SEED} \
  --log_dir logs/direction2/seed_${SEED} \
  --seed ${SEED}


