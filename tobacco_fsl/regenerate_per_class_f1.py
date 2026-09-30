"""
regenerate_per_class_f1.py — Standalone utility to regenerate per_class_f1.png
from an existing results/per_class_metrics.csv, using the corrected
(run-averaged) plotting logic in visualization.py.

Run this once after updating visualization.py, without re-running the full
experiment. Reads results/per_class_metrics.csv and overwrites
results/per_class_f1.png in place.

Usage:
    python regenerate_per_class_f1.py
    python regenerate_per_class_f1.py --k-shot 10
    python regenerate_per_class_f1.py --results-dir path\to\results
"""

import argparse
import os

import pandas as pd

import config
import visualization


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate per_class_f1.png from existing CSV")
    parser.add_argument("--results-dir", type=str, default=config.RESULTS_DIR,
                         help="Folder containing per_class_metrics.csv (default: config.RESULTS_DIR)")
    parser.add_argument("--k-shot", type=int, default=10,
                         help="k-shot value to plot (default: 10)")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = os.path.join(args.results_dir, "per_class_metrics.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Check --results-dir.")

    per_class_df = pd.read_csv(csv_path)

    # per_class_metrics.csv stores k_shot as float for the few-shot methods
    # (e.g. 10.0) and the string "N/A" for full_finetune rows; make sure the
    # requested k_shot matches how it's stored.
    available_k = sorted(per_class_df.loc[per_class_df["k_shot"].notna(), "k_shot"].unique())
    if args.k_shot not in [int(float(k)) for k in available_k]:
        raise ValueError(f"k_shot={args.k_shot} not found in {csv_path}. Available: {available_k}")

    visualization.plot_per_class_f1(per_class_df, k_shot=args.k_shot, class_names=config.CLASS_NAMES)

    out_path = os.path.join(args.results_dir, "per_class_f1.png")
    print(f"per_class_f1.png regenerated (k={args.k_shot}) at: {out_path}")


if __name__ == "__main__":
    main()
