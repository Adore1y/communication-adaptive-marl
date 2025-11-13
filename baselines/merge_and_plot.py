#!/usr/bin/env python3
import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=[
        "marl/baselines/results_qmix.csv",
        "marl/baselines/results_maddpg.csv",
        "marl/baselines/results_ppo.csv",
    ])
    ap.add_argument("--out_csv", default="marl/baselines/results_all.csv")
    ap.add_argument("--out_fig", default="marl/baselines/results_bar.png")
    args = ap.parse_args()

    frames = []
    for p in args.inputs:
        for f in glob.glob(p):
            if os.path.exists(f):
                frames.append(pd.read_csv(f))
    if not frames:
        print("No inputs found.")
        return
    df = pd.concat(frames, ignore_index=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Saved merged:", args.out_csv)

    # Basic plot: episode_reward_mean by algo (last row per algo)
    last = df.sort_values("total_steps").groupby(["scenario","algo"], as_index=False).last()
    plt.figure(figsize=(7,4))
    for scenario, g in last.groupby("scenario"):
        plt.bar(g["algo"], g["episode_reward_mean"], alpha=0.6, label=scenario)
    plt.xticks(rotation=30, ha='right')
    plt.ylabel("Episode Reward Mean")
    plt.title("Baseline Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150)
    print("Saved figure:", args.out_fig)


if __name__ == "__main__":
    main()

