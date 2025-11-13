#!/usr/bin/env python3
import argparse
import csv


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    rows = load_rows(args.results)
    by_algo = {}

    for row in rows:
        algo = row["algo"]
        by_algo.setdefault(algo, []).append(row)

    print("Algo, Success@last, Return@last, Len@last, N")
    for algo, records in by_algo.items():
        last = records[-1]
        print(
            f"{algo},{last['success_rate']},{last['avg_return']},{last['avg_length']},{len(records)}"
        )


