"""Aggregate measured per-run metrics into mean and standard deviation summaries."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="summary.csv produced by a training script")
    p.add_argument("--output", default="runs/aggregate.csv")
    args = p.parse_args()
    df = pd.read_csv(args.input)
    group = [c for c in ("model", "held_out_species") if c in df]
    measures = [c for c in ("accuracy", "macro_f1", "ordinal_mae", "safety_far",
                             "coverage", "accepted_safety_far") if c in df]
    if not measures:
        raise ValueError("No known metric columns found")
    if group:
        result = df.groupby(group, dropna=False)[measures].agg(["mean", "std"])
    else:
        result = df[measures].agg(["mean", "std"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output)
    print(result.to_string())

if __name__ == "__main__":
    main()
