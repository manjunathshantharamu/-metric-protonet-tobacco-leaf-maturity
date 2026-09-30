"""
visualization.py — Publication-ready plots (300 DPI), saved to config.RESULTS_DIR.

All plotting uses matplotlib's non-interactive Agg backend so this module can
run headlessly (CI, remote servers, etc.).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

import config

rcParams["figure.dpi"] = 300
rcParams["savefig.dpi"] = 300
rcParams["font.size"] = 10
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
rcParams["axes.linewidth"] = 0.8
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3

METHOD_LABELS = {
    "transfer_learning": "Transfer\nLearning",
    "protonet": "ProtoNet",
    "metric_protonet": "Metric-ProtoNet\n(Proposed)",
    "full_finetune": "Full\nFine-Tune",
}

METHOD_COLORS = {
    "transfer_learning": "#1f77b4",  # blue
    "protonet": "#ff7f0e",           # orange
    "metric_protonet": "#2ca02c",    # green
    "full_finetune": "#9467bd",      # purple
}


def _outpath(filename: str) -> str:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    return os.path.join(config.RESULTS_DIR, filename)


def plot_accuracy_vs_kshot(results_df, few_shot_methods=None, full_finetune_acc=None):
    """Line plot: x=k-shot, y=mean accuracy, one line per method, with std error bars."""
    if few_shot_methods is None:
        few_shot_methods = [m for m in config.METHODS if m != "full_finetune"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for method in few_shot_methods:
        sub = results_df[results_df["method"] == method]
        grouped = sub.groupby("k_shot")["accuracy"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            grouped["k_shot"], grouped["mean"] * 100, yerr=grouped["std"] * 100,
            label=METHOD_LABELS.get(method, method).replace("\n", " "),
            color=METHOD_COLORS.get(method), marker="o", capsize=3, linewidth=1.5,
        )

    if full_finetune_acc is not None:
        ax.axhline(full_finetune_acc * 100, color=METHOD_COLORS["full_finetune"],
                    linestyle="--", linewidth=1.5, label="Full Fine-Tune (upper bound)")

    ax.set_xlabel("k-shot")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. k-shot")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(_outpath("accuracy_vs_kshot.png"))
    plt.close(fig)


def plot_confusion_matrix(cm, method, k_shot, class_names=None):
    """Heatmap confusion matrix for one method at one k-shot, annotated with
    counts and percentages."""
    if class_names is None:
        class_names = config.CLASS_NAMES

    cm = np.asarray(cm)
    cm_pct = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                     ha="center", va="center",
                     color="white" if cm_pct[i, j] > 50 else "black", fontsize=8)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{METHOD_LABELS.get(method, method).replace(chr(10), ' ')} (k={k_shot})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(_outpath(f"confusion_matrix_{method}_k{k_shot}.png"))
    plt.close(fig)


def plot_tsne_features(features, labels, class_names=None):
    """t-SNE of the feature space, colored by true class."""
    from sklearn.manifold import TSNE

    if class_names is None:
        class_names = config.CLASS_NAMES

    n_samples = len(features)
    perplexity = min(30, max(5, n_samples // 4))

    tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED, perplexity=perplexity, init="pca")
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(6, 5))
    for c, name in enumerate(class_names):
        mask = labels == c
        ax.scatter(embedded[mask, 0], embedded[mask, 1], label=name, alpha=0.7, s=20)

    ax.set_title("t-SNE of Feature Space")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_outpath("tsne_features.png"))
    plt.close(fig)


def plot_per_class_f1(per_class_df, k_shot=10, class_names=None):
    """Grouped bar chart: per-class F1 for each method at a given k-shot."""
    if class_names is None:
        class_names = config.CLASS_NAMES

    sub = per_class_df[per_class_df["k_shot"] == k_shot]
    methods = [m for m in config.METHODS if m in sub["method"].unique()]

    x = np.arange(len(class_names))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, method in enumerate(methods):
        method_sub = sub[sub["method"] == method]
        # Average across all runs for each class (method_sub may contain
        # multiple rows per class, one per run) rather than taking the
        # first row, which would reflect only a single run's noise.
        f1_vals = [
            method_sub[method_sub["class"] == cname]["f1"].mean()
            if len(method_sub[method_sub["class"] == cname]) else 0.0
            for cname in class_names
        ]
        ax.bar(x + i * width, f1_vals, width, label=METHOD_LABELS.get(method, method).replace("\n", " "),
               color=METHOD_COLORS.get(method))

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("F1 score")
    ax.set_title(f"Per-Class F1 by Method (k={k_shot})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_outpath("per_class_f1.png"))
    plt.close(fig)


def plot_box_accuracy(results_df, k_shot, few_shot_methods=None):
    """Box plot of accuracy distribution across runs, per method, at a given k-shot."""
    if few_shot_methods is None:
        few_shot_methods = [m for m in config.METHODS if m != "full_finetune"]

    sub = results_df[results_df["k_shot"] == k_shot]
    data = [sub[sub["method"] == m]["accuracy"].values * 100 for m in few_shot_methods]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticks(range(1, len(few_shot_methods) + 1))
    ax.set_xticklabels([METHOD_LABELS.get(m, m).replace("\n", " ") for m in few_shot_methods])
    for patch, method in zip(bp["boxes"], few_shot_methods):
        patch.set_facecolor(METHOD_COLORS.get(method, "#cccccc"))
        patch.set_alpha(0.6)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy Distribution Across Runs (k={k_shot})")
    fig.tight_layout()
    fig.savefig(_outpath(f"box_accuracy_{k_shot}.png"))
    plt.close(fig)


def plot_metric_weights_distribution(weights, k_shot=10):
    """Histogram of learned metric weights for Metric ProtoNet."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(weights, bins=40, color=METHOD_COLORS["metric_protonet"], alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="w=1.0 (uniform/ProtoNet)")
    ax.set_xlabel("Learned weight value")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Learned Metric Weights (k={k_shot})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_outpath("metric_weights_distribution.png"))
    plt.close(fig)
