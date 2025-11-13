#!/usr/bin/env python3
"""统计通信与性能等指标（兼容反思/工具列）"""

import csv
import argparse


FIELDS = [
    "scenario", "algo", "success_rate", "avg_time",
    "comm_budget", "comm_usage", "robustness",
    "tool_calls", "tool_success", "reflection_rate", "reflection_success",
    "notes",
]


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="CSV with flexible columns")
    args = parser.parse_args()

    rows = load_rows(args.results)
    print(",".join(FIELDS))
    for r in rows:
        out = []
        for k in FIELDS:
            out.append(r.get(k, ""))
        print(",".join(out))
