#!/usr/bin/env python3
"""
Add lane labels to a trajectory CSV.

If no breakpoints are provided, the script will cluster the x-centers
into four groups and derive the three boundaries automatically.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DEFAULT_LABELS = ["non-motor", "lan1", "lan2", "lan3"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append a lane column to a trajectory CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV (e.g. data/processed/trajectories_raw_video1.csv)",
    )
    parser.add_argument(
        "--output",
        help="Path to output CSV. Omit to overwrite the input file.",
    )
    parser.add_argument(
        "--column-name",
        default="lane",
        help="Name of the new column (default: lane)",
    )
    parser.add_argument(
        "--breakpoints",
        type=float,
        nargs=3,
        metavar=("B1", "B2", "B3"),
        help="Three pixel breakpoints separating the four lanes. "
        "If omitted, boundaries are inferred automatically.",
    )
    return parser.parse_args()


def build_boundaries(breakpoints):
    return [-float("inf"), *breakpoints, float("inf")]


def assign_lane(value, boundaries, labels):
    if pd.isna(value):
        return None
    for idx, label in enumerate(labels):
        if boundaries[idx] <= value < boundaries[idx + 1]:
            return label
    return labels[-1]


def infer_breakpoints(x_values, n_clusters=4):
    valid = np.asarray(x_values, dtype=float)
    valid = valid[~np.isnan(valid)]
    if len(valid) < n_clusters:
        raise ValueError("Not enough valid x values to infer lanes.")
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    centers = np.sort(model.fit(valid.reshape(-1, 1)).cluster_centers_.flatten())
    return [(a + b) / 2 for a, b in zip(centers[:-1], centers[1:])]


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if "x" not in df.columns:
        raise ValueError("Input CSV must contain an 'x' column.")

    if args.breakpoints:
        if len(args.breakpoints) != len(DEFAULT_LABELS) - 1:
            raise ValueError("Provide exactly three breakpoints for four lanes.")
        breakpoints = sorted(args.breakpoints)
        source = "user-provided"
    else:
        breakpoints = infer_breakpoints(df["x"].values)
        source = "auto-inferred"

    boundaries = build_boundaries(breakpoints)
    df[args.column_name] = df["x"].apply(
        lambda val: assign_lane(val, boundaries, DEFAULT_LABELS)
    )

    output_path = Path(args.output) if args.output else input_path
    df.to_csv(output_path, index=False)
    print(
        f"Lane column '{args.column_name}' added using {source} breakpoints {breakpoints}.\n"
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
