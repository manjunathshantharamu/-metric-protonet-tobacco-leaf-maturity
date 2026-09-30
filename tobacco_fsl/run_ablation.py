"""
run_ablation.py — Ablation study for the Metric ProtoNet's learnable diagonal metric.

Experiments:
    A1: Uniform weights (w=1.0, frozen)       -> should equal standard ProtoNet
    A2: Random frozen weights (untrained)      -> shows that LEARNING matters, not
                                                   just any non-uniform weighting
    A3: Weight decay sweep (lambda = 0, 0.01, 0.1, 1.0)
    A4: Learning rate sweep (lr = 0.001, 0.01, 0.1)

Protocol matches the main experiment: 5 k-shot settings, 10 independent runs
per setting, frozen MobileNetV3-Small backbone, same train/val/test split.

Usage:
    python run_ablation.py
    python run_ablation.py --quick
    python run_ablation.py --dataset /path
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import data_loader
from feature_extractor import FeatureExtractor
from methods import protonet, metric_protonet


def parse_args():
    parser = argparse.ArgumentParser(description="Metric ProtoNet Ablation Study")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    return parser.parse_args()


def run_ablation_a1_a2(extractor, train_files, train_labels, k_shots, n_runs, results_dir):
    """A1: uniform frozen (w=1.0) vs learned. A2: random frozen (N(1.0, 0.1)) vs learned."""
    rows = []

    for k_shot in k_shots:
        for run_idx in range(n_runs):
            seed = config.RANDOM_SEED + run_idx * 100 + k_shot
            support_files, support_labels, query_files, query_labels = data_loader.sample_episode(
                train_files, train_labels, k_shot, config.N_CLASSES, seed
            )
            support_images = data_loader.load_images(support_files, config.IMAGE_SIZE)
            query_images = data_loader.load_images(query_files, config.IMAGE_SIZE)
            support_feats = extractor.extract_batch(support_images)
            query_feats = extractor.extract_batch(query_images)

            D = support_feats.shape[1]

            # A1: uniform frozen (equivalent to standard ProtoNet)
            uniform_metrics = metric_protonet.run_metric_protonet(
                support_feats, support_labels, query_feats, query_labels,
                n_classes=config.N_CLASSES,
                init_weights=np.ones(D), freeze_weights=True,
            )

            # A2: random frozen weights ~ N(1.0, 0.1)
            rng = np.random.RandomState(seed)
            random_weights = rng.normal(loc=1.0, scale=0.1, size=D).astype(np.float32)
            random_metrics = metric_protonet.run_metric_protonet(
                support_feats, support_labels, query_feats, query_labels,
                n_classes=config.N_CLASSES,
                init_weights=random_weights, freeze_weights=True,
            )

            # Learned (normal Metric ProtoNet training)
            learned_metrics = metric_protonet.run_metric_protonet(
                support_feats, support_labels, query_feats, query_labels,
                n_classes=config.N_CLASSES,
            )

            # Reference: standard ProtoNet
            proto_metrics = protonet.run_protonet(
                support_feats, support_labels, query_feats, query_labels,
                n_classes=config.N_CLASSES,
            )

            rows.append({"experiment": "A1_uniform_frozen", "k_shot": k_shot, "run": run_idx,
                         "accuracy": uniform_metrics["accuracy"]})
            rows.append({"experiment": "A1_learned", "k_shot": k_shot, "run": run_idx,
                         "accuracy": learned_metrics["accuracy"]})
            rows.append({"experiment": "A1_reference_protonet", "k_shot": k_shot, "run": run_idx,
                         "accuracy": proto_metrics["accuracy"]})
            rows.append({"experiment": "A2_random_frozen", "k_shot": k_shot, "run": run_idx,
                         "accuracy": random_metrics["accuracy"]})
            rows.append({"experiment": "A2_learned", "k_shot": k_shot, "run": run_idx,
                         "accuracy": learned_metrics["accuracy"]})

    return pd.DataFrame(rows)


def run_ablation_a3(extractor, train_files, train_labels, k_shots, n_runs):
    """A3: weight decay sweep, all other hyperparameters fixed."""
    weight_decays = [0.0, 0.01, 0.1, 1.0]
    rows = []

    for k_shot in k_shots:
        for run_idx in range(n_runs):
            seed = config.RANDOM_SEED + run_idx * 100 + k_shot
            support_files, support_labels, query_files, query_labels = data_loader.sample_episode(
                train_files, train_labels, k_shot, config.N_CLASSES, seed
            )
            support_images = data_loader.load_images(support_files, config.IMAGE_SIZE)
            query_images = data_loader.load_images(query_files, config.IMAGE_SIZE)
            support_feats = extractor.extract_batch(support_images)
            query_feats = extractor.extract_batch(query_images)

            for wd in weight_decays:
                metrics = metric_protonet.run_metric_protonet(
                    support_feats, support_labels, query_feats, query_labels,
                    n_classes=config.N_CLASSES, weight_decay=wd,
                )
                rows.append({"weight_decay": wd, "k_shot": k_shot, "run": run_idx,
                             "accuracy": metrics["accuracy"]})

    return pd.DataFrame(rows)


def run_ablation_a4(extractor, train_files, train_labels, k_shots, n_runs):
    """A4: learning rate sweep, all other hyperparameters fixed."""
    learning_rates = [0.001, 0.01, 0.1]
    rows = []

    for k_shot in k_shots:
        for run_idx in range(n_runs):
            seed = config.RANDOM_SEED + run_idx * 100 + k_shot
            support_files, support_labels, query_files, query_labels = data_loader.sample_episode(
                train_files, train_labels, k_shot, config.N_CLASSES, seed
            )
            support_images = data_loader.load_images(support_files, config.IMAGE_SIZE)
            query_images = data_loader.load_images(query_files, config.IMAGE_SIZE)
            support_feats = extractor.extract_batch(support_images)
            query_feats = extractor.extract_batch(query_images)

            for lr in learning_rates:
                metrics = metric_protonet.run_metric_protonet(
                    support_feats, support_labels, query_feats, query_labels,
                    n_classes=config.N_CLASSES, lr=lr,
                )
                rows.append({"lr": lr, "k_shot": k_shot, "run": run_idx,
                             "accuracy": metrics["accuracy"]})

    return pd.DataFrame(rows)


def plot_a1(df, results_dir):
    grouped = df[df["experiment"].isin(["A1_uniform_frozen", "A1_learned"])].groupby(
        ["experiment", "k_shot"])["accuracy"].mean().unstack(level=0) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped.T.plot(kind="bar", ax=ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A1: Uniform Frozen vs Learned Weights")
    ax.set_xlabel("k-shot")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "ablation_a1_uniform_vs_learned.png"))
    plt.close(fig)


def plot_a2(df, results_dir):
    grouped = df[df["experiment"].isin(["A2_random_frozen", "A2_learned"])].groupby(
        ["experiment", "k_shot"])["accuracy"].mean().unstack(level=0) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped.T.plot(kind="bar", ax=ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A2: Random Frozen vs Learned Weights")
    ax.set_xlabel("k-shot")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "ablation_a2_random_vs_learned.png"))
    plt.close(fig)


def plot_a3(df, results_dir):
    grouped = df.groupby(["weight_decay", "k_shot"])["accuracy"].mean().unstack(level=0) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    for wd in grouped.columns:
        ax.plot(grouped.index, grouped[wd], marker="o", label=f"lambda={wd}")
    ax.set_xlabel("k-shot")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A3: Weight Decay Sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "ablation_a3_weight_decay.png"))
    plt.close(fig)


def plot_a4(df, results_dir):
    grouped = df.groupby(["lr", "k_shot"])["accuracy"].mean().unstack(level=0) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    for lr in grouped.columns:
        ax.plot(grouped.index, grouped[lr], marker="o", label=f"lr={lr}")
    ax.set_xlabel("k-shot")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A4: Learning Rate Sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "ablation_a4_learning_rate.png"))
    plt.close(fig)


def main():
    args = parse_args()
    dataset_path = args.dataset or config.DATASET_PATH
    results_dir = args.results_dir or config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    k_shots = [10] if args.quick else config.K_SHOTS
    n_runs = 1 if args.quick else config.N_RUNS

    filepaths, labels_str, labels_idx = data_loader.load_dataset(dataset_path)
    splits = data_loader.stratified_split(
        filepaths, labels_idx, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE, seed=config.RANDOM_SEED
    )
    train_files, train_labels = splits["train"]

    extractor = FeatureExtractor(backbone=config.BACKBONE, pretrained=config.PRETRAINED, device=config.DEVICE)

    print("Running A1/A2 (uniform / random / learned weights)...")
    a1_a2_df = run_ablation_a1_a2(extractor, train_files, train_labels, k_shots, n_runs, results_dir)

    print("Running A3 (weight decay sweep)...")
    a3_df = run_ablation_a3(extractor, train_files, train_labels, k_shots, n_runs)

    print("Running A4 (learning rate sweep)...")
    a4_df = run_ablation_a4(extractor, train_files, train_labels, k_shots, n_runs)

    a1_a2_df.to_csv(os.path.join(results_dir, "ablation_results.csv"), index=False, encoding="utf-8")
    a3_df.to_csv(os.path.join(results_dir, "ablation_a3_results.csv"), index=False, encoding="utf-8")
    a4_df.to_csv(os.path.join(results_dir, "ablation_a4_results.csv"), index=False, encoding="utf-8")

    summary = {
        "A1_uniform_frozen_mean": a1_a2_df[a1_a2_df["experiment"] == "A1_uniform_frozen"]["accuracy"].mean(),
        "A1_learned_mean": a1_a2_df[a1_a2_df["experiment"] == "A1_learned"]["accuracy"].mean(),
        "A1_reference_protonet_mean": a1_a2_df[a1_a2_df["experiment"] == "A1_reference_protonet"]["accuracy"].mean(),
        "A2_random_frozen_mean": a1_a2_df[a1_a2_df["experiment"] == "A2_random_frozen"]["accuracy"].mean(),
        "A2_learned_mean": a1_a2_df[a1_a2_df["experiment"] == "A2_learned"]["accuracy"].mean(),
    }
    for wd in a3_df["weight_decay"].unique():
        summary[f"A3_wd_{wd}_mean"] = a3_df[a3_df["weight_decay"] == wd]["accuracy"].mean()
    for lr in a4_df["lr"].unique():
        summary[f"A4_lr_{lr}_mean"] = a4_df[a4_df["lr"] == lr]["accuracy"].mean()

    pd.DataFrame([summary]).to_csv(os.path.join(results_dir, "ablation_summary.csv"), index=False, encoding="utf-8")

    plot_a1(a1_a2_df, results_dir)
    plot_a2(a1_a2_df, results_dir)
    plot_a3(a3_df, results_dir)
    plot_a4(a4_df, results_dir)

    print("\nAblation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v*100:.2f}%")
    print(f"\nAblation results saved to: {results_dir}")


if __name__ == "__main__":
    main()
