"""
generate_results_report.py — Multi-sheet Excel report + additional publication plots.

Reads CSVs from results/ (produced by run_experiment.py and run_ablation.py)
and generates:

    results/experiment_results.xlsx — 7 sheets:
        1. Summary            method x k-shot, mean +/- std accuracy
        2. Raw_Results         all individual run results
        3. Pivot_Mean          methods (rows) x k-shot (cols), mean accuracy (%)
        4. Pivot_Std           same, but standard deviation
        5. Per_Class_Metrics   per-class precision/recall/F1
        6. Statistical_Tests   Wilcoxon p-values
        7. Best_Per_K          best method at each k-shot

    Additional plots:
        heatmap_accuracy.png, radar_per_class.png, bar_accuracy_gain.png,
        box_accuracy_distribution.png, statistical_significance_k{k}.png,
        summary_table.png

Usage:
    python generate_results_report.py
    python generate_results_report.py --results_dir path
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

METHOD_ORDER = ["transfer_learning", "protonet", "metric_protonet", "full_finetune"]
METHOD_DISPLAY = {
    "transfer_learning": "Transfer Learning",
    "protonet": "ProtoNet",
    "metric_protonet": "Metric-ProtoNet (Proposed)",
    "full_finetune": "Full Fine-Tune",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Excel report and additional plots")
    parser.add_argument("--results_dir", type=str, default=config.RESULTS_DIR)
    return parser.parse_args()


def load_csvs(results_dir):
    def _read(name):
        path = os.path.join(results_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    return {
        "results": _read("results_summary.csv"),
        "per_class": _read("per_class_metrics.csv"),
        "stats": _read("statistical_tests.csv"),
    }


def build_excel_report(data, results_dir):
    results_df = data["results"]
    per_class_df = data["per_class"]
    stats_df = data["stats"]

    if results_df.empty:
        print("No results_summary.csv found — skipping Excel report.")
        return

    summary = results_df.groupby(["method", "k_shot"])["accuracy"].agg(["mean", "std"]).reset_index()
    summary["mean"] *= 100
    summary["std"] *= 100
    summary.columns = ["method", "k_shot", "mean_accuracy_pct", "std_accuracy_pct"]

    pivot_mean = results_df.groupby(["method", "k_shot"])["accuracy"].mean().unstack() * 100
    pivot_std = results_df.groupby(["method", "k_shot"])["accuracy"].std().unstack() * 100

    best_per_k = (
        results_df.groupby(["k_shot", "method"])["accuracy"].mean()
        .reset_index()
        .sort_values("accuracy", ascending=False)
        .groupby("k_shot")
        .first()
        .reset_index()
    )
    best_per_k["accuracy_pct"] = best_per_k["accuracy"] * 100

    out_path = os.path.join(results_dir, "experiment_results.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        results_df.to_excel(writer, sheet_name="Raw_Results", index=False)
        pivot_mean.to_excel(writer, sheet_name="Pivot_Mean")
        pivot_std.to_excel(writer, sheet_name="Pivot_Std")
        per_class_df.to_excel(writer, sheet_name="Per_Class_Metrics", index=False)
        stats_df.to_excel(writer, sheet_name="Statistical_Tests", index=False)
        best_per_k.to_excel(writer, sheet_name="Best_Per_K", index=False)

    print(f"Excel report written to: {out_path}")
    return pivot_mean, pivot_std


def plot_heatmap_accuracy(pivot_mean, results_dir):
    methods = [m for m in METHOD_ORDER if m in pivot_mean.index]
    data = pivot_mean.loc[methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data.values, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in methods])
    ax.set_xlabel("k-shot")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    ax.set_title("Mean Accuracy Heatmap (%)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "heatmap_accuracy.png"))
    plt.close(fig)


def plot_radar_per_class(per_class_df, results_dir, k_shot=10):
    sub = per_class_df[per_class_df["k_shot"] == k_shot]
    if sub.empty:
        return
    methods = [m for m in METHOD_ORDER if m in sub["method"].unique()]
    class_names = config.CLASS_NAMES

    angles = np.linspace(0, 2 * np.pi, len(class_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for method in methods:
        method_sub = sub[sub["method"] == method]
        # Average across all runs for each class rather than taking the
        # first row (method_sub has one row per run per class).
        values = [
            method_sub[method_sub["class"] == c]["f1"].mean()
            if len(method_sub[method_sub["class"] == c]) else 0.0
            for c in class_names
        ]
        values += values[:1]
        ax.plot(angles, values, label=METHOD_DISPLAY.get(method, method), linewidth=1.5)
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names)
    ax.set_title(f"Per-Class F1 Radar (k={k_shot})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "radar_per_class.png"))
    plt.close(fig)


def plot_bar_accuracy_gain(pivot_mean, results_dir):
    if "protonet" not in pivot_mean.index:
        return
    baseline = pivot_mean.loc["protonet"]
    methods = [m for m in METHOD_ORDER if m in pivot_mean.index and m != "protonet"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(pivot_mean.columns))
    width = 0.8 / max(len(methods), 1)

    for i, method in enumerate(methods):
        gain = pivot_mean.loc[method] - baseline
        ax.bar(x + i * width, gain, width, label=METHOD_DISPLAY.get(method, method))

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(pivot_mean.columns)
    ax.set_xlabel("k-shot")
    ax.set_ylabel("Accuracy gain over ProtoNet (pct. points)")
    ax.set_title("Accuracy Gain Over Standard ProtoNet")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "bar_accuracy_gain.png"))
    plt.close(fig)


def plot_box_accuracy_distribution(results_df, results_dir):
    methods = [m for m in METHOD_ORDER if m in results_df["method"].unique() and m != "full_finetune"]
    k_shots = sorted(results_df["k_shot"].unique())

    fig, axes = plt.subplots(1, len(k_shots), figsize=(4 * len(k_shots), 4.5), sharey=True)
    if len(k_shots) == 1:
        axes = [axes]

    for ax, k in zip(axes, k_shots):
        sub = results_df[results_df["k_shot"] == k]
        data = [sub[sub["method"] == m]["accuracy"].values * 100 for m in methods]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in methods])
        ax.set_title(f"k={k}")
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Accuracy Distribution Across Runs")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "box_accuracy_distribution.png"))
    plt.close(fig)


def plot_statistical_significance(stats_df, results_dir):
    if stats_df.empty:
        return

    for k in sorted(stats_df["k_shot"].unique()):
        sub = stats_df[stats_df["k_shot"] == k]
        comparisons = sub["comparison"].tolist()
        p_values = sub["p_value"].values

        fig, ax = plt.subplots(figsize=(6, max(2, 0.5 * len(comparisons))))
        colors = ["#2ca02c" if p < 0.01 else "#ff7f0e" if p < 0.05 else "#d62728" for p in p_values]
        ax.barh(comparisons, [1] * len(comparisons), color=colors)
        for i, p in enumerate(p_values):
            ax.text(0.5, i, f"p={p:.4f}", ha="center", va="center", fontsize=8, color="white")
        ax.set_xticks([])
        ax.set_title(f"Statistical Significance (Wilcoxon, k={k})\ngreen=p<0.01, orange=p<0.05, red=ns")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"statistical_significance_k{k}.png"))
        plt.close(fig)


def plot_summary_table(pivot_mean, pivot_std, results_dir):
    methods = [m for m in METHOD_ORDER if m in pivot_mean.index]

    cell_text = []
    for method in methods:
        row = []
        for k in pivot_mean.columns:
            m = pivot_mean.loc[method, k]
            s = pivot_std.loc[method, k] if method in pivot_std.index else np.nan
            if np.isnan(m):
                row.append("-")
            elif np.isnan(s):
                row.append(f"{m:.1f}")
            else:
                row.append(f"{m:.1f}\u00b1{s:.1f}")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(2 + 1.3 * len(pivot_mean.columns), 1 + 0.5 * len(methods)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=[METHOD_DISPLAY.get(m, m) for m in methods],
        colLabels=[f"k={k}" for k in pivot_mean.columns],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    if "metric_protonet" in methods:
        row_idx = methods.index("metric_protonet") + 1  # +1 for header row
        for col_idx in range(len(pivot_mean.columns)):
            table[(row_idx, col_idx)].set_facecolor("#d9f2d9")

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "summary_table.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    data = load_csvs(results_dir)
    if data["results"].empty:
        print(f"No results found in {results_dir}. Run run_experiment.py first.")
        return

    pivot_mean, pivot_std = build_excel_report(data, results_dir)

    plot_heatmap_accuracy(pivot_mean, results_dir)
    plot_radar_per_class(data["per_class"], results_dir, k_shot=10 if 10 in data["results"]["k_shot"].values else data["results"]["k_shot"].iloc[0])
    plot_bar_accuracy_gain(pivot_mean, results_dir)
    plot_box_accuracy_distribution(data["results"], results_dir)
    plot_statistical_significance(data["stats"], results_dir)
    plot_summary_table(pivot_mean, pivot_std, results_dir)

    print(f"Additional plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
