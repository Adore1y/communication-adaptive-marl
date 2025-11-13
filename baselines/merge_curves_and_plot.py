#!/usr/bin/env python3
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    mu = np.nanmean(x)
    se = np.nanstd(x, ddof=1) / max(1, np.sqrt(n))
    return mu, 1.96 * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=[
        "marl/baselines/curves_*.csv",
    ])
    ap.add_argument("--out_csv", default="marl/baselines/curves_merged.csv")
    ap.add_argument("--out_fig", default="marl/baselines/curves_ci.png")
    args = ap.parse_args()

    frames = []
    for pat in args.inputs:
        for f in glob.glob(pat):
            if os.path.exists(f):
                frames.append(pd.read_csv(f))
    if not frames:
        print("No curve inputs found.")
        return
    df = pd.concat(frames, ignore_index=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # group by scenario, algo, step; aggregate across seeds
    groups = df.groupby(["scenario","algo","step"]) 
    rows = []
    for (sc, algo, step), g in groups:
        mu, half = ci95(g["reward"].values)
        rows.append({"scenario": sc, "algo": algo, "step": step, "mean": mu, "ci": half})
    agg = pd.DataFrame(rows).sort_values(["scenario","algo","step"]).reset_index(drop=True)
    agg.to_csv(args.out_csv, index=False)
    print("Saved merged curves:", args.out_csv)

    # plot
    plt.figure(figsize=(8,5))
    for (sc, algo), g in agg.groupby(["scenario","algo"]):
        x = g["step"].values
        y = g["mean"].values
        ci = g["ci"].values
        plt.plot(x, y, label=f"{algo}-{sc}")
        plt.fill_between(x, y-ci, y+ci, alpha=0.15)
    plt.xlabel("Steps")
    plt.ylabel("Episode Reward Mean")
    plt.title("Learning Curves (mean ± 95% CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150)
    print("Saved figure:", args.out_fig)


if __name__ == "__main__":
    main()

