#!/usr/bin/env python3
"""Agentic RL 训练入口：记忆-反思-工具钩子（含反思频率与工具预算、结构化日志）"""

import argparse
import csv
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


ENV_MAP = {
    "minigrid_doorkey": lambda: gym.make("MiniGrid-DoorKey-5x5-v0", render_mode=None)
}


class EpisodicMemory:
    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.buffer = []

    def add(self, traj):
        self.buffer.append(traj)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, k: int = 8):
        if not self.buffer:
            return []
        idx = np.random.choice(len(self.buffer), size=min(k, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in idx]


class ReflectionModule:
    def summarize_failures(self, traj):
        # 占位实现——真实项目可接LLM或启发式
        return "avoid dead-ends; remember key; shorten detours."


class ToolBox:
    def plan(self, goal_desc: str, observation_desc: str):
        # 占位实现：返回子目标列表
        return {"subgoals": ["reach key", "open door", "reach goal"]}


class AgenticMetricsCallback(BaseCallback):
    """在训练期间统计反思与工具调用指标，并周期性写入CSV。"""

    def __init__(self, log_dir: str, eval_interval: int, reflection_interval: int,
                 tool_budget_per_10k: int, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.eval_interval = eval_interval
        self.reflection_interval = max(1, reflection_interval)
        self.tool_budget_per_10k = max(0, tool_budget_per_10k)

        self.tool_calls_total = 0
        self.tool_success_total = 0
        self.reflection_triggers_total = 0
        self.reflection_success_total = 0

        self._csv_path = self.log_dir / "metrics_agentic.csv"
        self._ensure_header()

    def _ensure_header(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "global_step", "tool_calls", "tool_success",
                    "reflection_triggers", "reflection_success"
                ])

    def _write_row(self, step: int):
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                self.tool_calls_total,
                self.tool_success_total,
                self.reflection_triggers_total,
                self.reflection_success_total,
            ])

    def on_step(self) -> bool:
        step = int(self.num_timesteps)
        # 反思触发：按反思间隔触发一次（占位策略）
        if step % self.reflection_interval == 0:
            self.reflection_triggers_total += 1
            # 占位：认为一次反思有一定成功率（这里简单计为成功）
            self.reflection_success_total += 1

        # 工具预算：每 10k 步允许调用 N 次（占位模拟）
        # 若当前步位于10k窗口起点，清零窗口内可用预算
        # 这里简化为：在 eval_interval 时模拟一次工具调用
        if self.eval_interval > 0 and step % self.eval_interval == 0:
            # 计算当前窗口预算
            window_start = (step // 10_000) * 10_000
            used_in_window = self.tool_calls_total - ((window_start // 10_000) * self.tool_budget_per_10k)
            remaining = self.tool_budget_per_10k - max(0, used_in_window)
            if remaining > 0:
                self.tool_calls_total += 1
                # 占位：工具调用成功一次
                self.tool_success_total += 1

        # 周期性落盘：与eval_interval对齐
        if self.eval_interval > 0 and step % self.eval_interval == 0:
            self._write_row(step)
        return True


def make_env(env_id: str):
    def thunk():
        return ENV_MAP[env_id]()
    return thunk


def train(args):
    env_fns = [make_env(args.env) for _ in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1)

    memory = EpisodicMemory(capacity=args.memory_capacity) if args.use_memory else None
    reflector = ReflectionModule() if args.use_reflection else None
    tools = ToolBox() if args.use_tools else None

    eval_cb = EvalCallback(
        vec_env,
        best_model_save_path=os.path.join(args.save_dir, "best"),
        log_path=os.path.join(args.log_dir, "eval"),
        eval_freq=args.eval_interval,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=args.eval_interval,
        save_path=args.save_dir,
        name_prefix="agentic",
    )
    metrics_cb = AgenticMetricsCallback(
        log_dir=args.log_dir,
        eval_interval=args.eval_interval,
        reflection_interval=args.reflection_interval,
        tool_budget_per_10k=args.tool_budget_per_10k,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_cb, ckpt_cb, metrics_cb],
        log_interval=10,
    )

    # 训练元信息
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.log_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "env": args.env,
                "algo": args.algo,
                "total_timesteps": args.total_timesteps,
                "use_memory": args.use_memory,
                "use_reflection": args.use_reflection,
                "use_tools": args.use_tools,
                "reflection_interval": args.reflection_interval,
                "tool_budget_per_10k": args.tool_budget_per_10k,
            },
            f,
        )

    vec_env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="minigrid_doorkey")
    parser.add_argument("--algo", default="PPO")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval_interval", type=int, default=50_000)
    parser.add_argument("--use_memory", action="store_true")
    parser.add_argument("--memory_capacity", type=int, default=10_000)
    parser.add_argument("--use_reflection", action="store_true")
    parser.add_argument("--use_tools", action="store_true")
    parser.add_argument("--reflection_interval", type=int, default=50_000,
                        help="反思触发的步频（steps）")
    parser.add_argument("--tool_budget_per_10k", type=int, default=5,
                        help="每10k步允许的工具调用预算")
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--log_dir", default="logs")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
