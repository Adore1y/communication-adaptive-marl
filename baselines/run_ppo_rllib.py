#!/usr/bin/env python3
import argparse, time
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from marl.baselines.utils_results import append_result_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="simple_spread_v3")
    ap.add_argument("--steps", type=int, default=5000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="marl/baselines/results_ppo.csv")
    ap.add_argument("--curve_out", default="")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    cfg = (PPOConfig()
           .environment(env=args.env)
           .rollouts(num_rollout_workers=args.workers)
           .resources(num_gpus=0)
           .framework("torch")
           .training(train_batch_size=4000, sgd_minibatch_size=1024, lr=3e-4)
           .debugging(seed=args.seed))

    algo = cfg.build()
    total_steps = 0
    last_mean = None
    try:
        while total_steps < args.steps:
            res = algo.train()
            total_steps = res.get("timesteps_total") or res.get("num_env_steps_trained", 0)
            last_mean = res.get("episode_reward_mean")
            if total_steps is None:
                total_steps = 0
            print(f"steps={total_steps} reward_mean={last_mean}")
            if args.curve_out:
                append_result_row(args.curve_out, {
                    "scenario": args.env,
                    "algo": "PPO(RLlib)",
                    "seed": args.seed,
                    "step": total_steps,
                    "reward": last_mean,
                }, ["scenario","algo","seed","step","reward"]) 
    finally:
        algo.stop()
        ray.shutdown()

    row = {
        "scenario": args.env,
        "algo": "PPO(RLlib)",
        "seed": args.seed,
        "total_steps": total_steps,
        "episode_reward_mean": last_mean,
        "notes": "rllib-ppo"
    }
    append_result_row(args.out, row,
                      ["scenario","algo","seed","total_steps","episode_reward_mean","notes"])
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
