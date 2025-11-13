#!/bin/bash
#SBATCH --job-name=multirobot_train
#SBATCH --nodes=4                  # 4个计算节点（示例）
#SBATCH --ntasks-per-node=8        # 每个节点8个任务
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2               # 每个节点2块GPU
#SBATCH --time=168:00:00           # 7天
#SBATCH --output=logs/multirobot_%j.out
#SBATCH --error=logs/multirobot_%j.err

# 大规模多智能体机器人协同学习训练（可按需改为单节点8卡）

echo "=== 多智能体机器人协同学习 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "======================================"

# 加载模块
module load python/3.9
module load cuda/11.8
module load openmpi/4.1.1

# 设置环境
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO

# 创建目录
mkdir -p logs results checkpoints

# 激活虚拟环境
source ~/venv/robot_env/bin/activate

# 安装依赖
pip install torch stable-baselines3[extra] pybullet gymnasium
pip install pettingzoo supersuit

# 分布式训练
srun python multirobot_training.py \
    --algorithm MAPPO \
    --num_agents 50 \
    --num_rollout_workers 32 \
    --lr 3e-4 \
    --total_timesteps 10000000 \
    --save_dir ./checkpoints \
    --eval_interval 100000

echo "训练完成！"


