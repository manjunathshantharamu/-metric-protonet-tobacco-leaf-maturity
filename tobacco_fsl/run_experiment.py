"""
run_experiment.py — Main experiment runner (CLI).

Usage:
    python run_experiment.py                                    # Full experiment
    python run_experiment.py --quick                             # Quick test (1 run, k=10 only)
    python run_experiment.py --dataset /path/to/data             # Custom dataset path
    python run_experiment.py --methods transfer_learning,protonet  # Subset of methods

Execution flow:
    1. Parse CLI arguments.
    2. Load dataset -> stratified split (train/val/test).
    3. Initialize the frozen feature extractor.
    4. For each k in K_SHOTS:
        a. For each run in range(N_RUNS):
            i.   Sample episode (support: k per class, query: remaining train)
            ii.  Extract features for support and query images
            iii. For each few-shot method: run it, record metrics
            iv.  Log progress
        b. Compute pairwise Wilcoxon signed-rank tests between all method
           pairs at this k-shot.
    5. Run Full Fine-Tune ONCE (uses all train+val data, evaluated on test set).
    6. Save all results to CSV files.
    7. Generate all visualizations.
    8. Print summary table.
"""

import argparse
import itertools
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config
import data_loader
import evaluation
import visualization
from feature_extractor import FeatureExtractor
from methods import transfer_learning, protonet, metric_protonet, full_finetune


METHOD_FUNCS = {
    "transfer_learning": transfer_learning.run_transfer_learning,
    "protonet": protonet.run_protonet,
    "metric_protonet": metric_protonet.run_metric_protonet,
}

METHOD_DISPLAY = {
    "transfer_learning": "Transfer Learning",
    "protonet": "ProtoNet",
    "metric_protonet": "Metric-ProtoNet",
    "full_finetune": "Full Fine-Tune",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Tobacco Leaf Few-Shot Classification Experiment")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    parser.add_argument("--quick", action="store_true", help="Quick test: 1 run, k=10 only")
    parser.add_argument("--methods", type=str, default=None,
                         help="Comma-separated subset of methods to run (e.g. transfer_learning,protonet)")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results output directory")
    return parser.parse_args()


def log(msg, log_file=None, verbose_level=1):
    if config.VERBOSE >= verbose_level:
        print(msg)
    if log_file is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {msg}\n")
        log_file.flush()


def main():
    args = parse_args()

    dataset_path = args.dataset or config.DATASET_PATH
    results_dir = args.results_dir or config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    k_shots = [10] if args.quick else config.K_SHOTS
    n_runs = 1 if args.quick else config.N_RUNS

    few_shot_methods = [m for m in config.METHODS if m != "full_finetune"]
    if args.methods:
        requested = args.methods.split(",")
        few_shot_methods = [m for m in requested if m != "full_finetune"]
    run_full_finetune_flag = ("full_finetune" in config.METHODS
                               and (not args.methods or "full_finetune" in args.methods.split(",")))

    log_path = os.path.join(results_dir, "experiment_log.txt")
    log_file = open(log_path, "a")

    log("=" * 60, log_file)
    log("  Tobacco Leaf Few-Shot Classification Experiment", log_file)
    log("=" * 60, log_file)

    # 1. Load dataset
    filepaths, labels_str, labels_idx = data_loader.load_dataset(dataset_path)
    splits = data_loader.stratified_split(
        filepaths, labels_idx, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE, seed=config.RANDOM_SEED
    )
    train_files, train_labels = splits["train"]
    val_files, val_labels = splits["val"]
    test_files, test_labels = splits["test"]

    log(f"Dataset: {len(filepaths)} images, {config.N_CLASSES} classes", log_file)
    log(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}", log_file)

    # 2. Feature extractor
    extractor = FeatureExtractor(backbone=config.BACKBONE, pretrained=config.PRETRAINED, device=config.DEVICE)
    log(f"Backend: PyTorch ({config.BACKBONE}, {extractor.feature_dim}-dim)", log_file)

    # 3. Main k-shot x run loop
    summary_rows = []
    per_class_rows = []
    statistical_rows = []
    metric_weights_at_k10 = None

    total_settings = len(k_shots)
    for k_i, k_shot in enumerate(k_shots):
        log(f"\n[{k_i+1}/{total_settings}] k={k_shot}", log_file)

        run_scores = {m: [] for m in few_shot_methods}
        last_run_cms = {}

        for run_idx in range(n_runs):
            seed = config.RANDOM_SEED + run_idx * 100 + k_shot

            support_files, support_labels, query_files, query_labels = data_loader.sample_episode(
                train_files, train_labels, k_shot, config.N_CLASSES, seed
            )

            support_images = data_loader.load_images(support_files, config.IMAGE_SIZE)
            query_images = data_loader.load_images(query_files, config.IMAGE_SIZE)

            support_feats = extractor.extract_batch(support_images)
            query_feats = extractor.extract_batch(query_images)

            log(f"  Run {run_idx+1}/{n_runs}", log_file, verbose_level=2)

            for method in few_shot_methods:
                func = METHOD_FUNCS[method]
                t0 = time.time()
                metrics = func(
                    support_feats, support_labels, query_feats, query_labels,
                    extractor, n_classes=config.N_CLASSES,
                )
                elapsed = time.time() - t0

                acc = metrics["accuracy"]
                run_scores[method].append(acc)

                summary_rows.append({
                    "method": method, "k_shot": k_shot, "run": run_idx,
                    "accuracy": acc,
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "top2_accuracy": metrics["top2_accuracy"],
                })

                for cname, stats in metrics["per_class"].items():
                    per_class_rows.append({
                        "method": method, "k_shot": k_shot, "class": cname,
                        "precision": stats["precision"], "recall": stats["recall"],
                        "f1": stats["f1"], "support": stats["support"],
                    })

                last_run_cms[method] = metrics["confusion_matrix"]

                if method == "metric_protonet" and k_shot == 10 and run_idx == n_runs - 1:
                    metric_weights_at_k10 = metrics["metric_weights"]

                log(f"    {METHOD_DISPLAY[method]:20s}: {acc*100:.2f}%  ({elapsed:.2f}s)",
                    log_file, verbose_level=2)

        # Per-k-shot summary line
        for method in few_shot_methods:
            scores = np.array(run_scores[method])
            log(f"  {METHOD_DISPLAY[method]:20s}: {scores.mean()*100:.2f}% +/- {scores.std()*100:.2f}%", log_file)

        # Confusion matrices for last run of this k-shot
        for method, cm in last_run_cms.items():
            visualization.plot_confusion_matrix(cm, method, k_shot)

        # Box plot for this k-shot
        results_df_partial = pd.DataFrame(summary_rows)
        visualization.plot_box_accuracy(results_df_partial, k_shot, few_shot_methods)

        # Pairwise Wilcoxon tests at this k-shot
        stat_rows_this_k = []
        for m1, m2 in itertools.combinations(few_shot_methods, 2):
            test_result = evaluation.wilcoxon_test(np.array(run_scores[m1]), np.array(run_scores[m2]))
            d = evaluation.cohens_d(np.array(run_scores[m1]), np.array(run_scores[m2]))
            stat_rows_this_k.append({
                "comparison": f"{m1} vs {m2}", "k_shot": k_shot,
                "p_value": test_result["p_value"], "statistic": test_result["statistic"],
                "significant": test_result["significant"], "cohens_d": d,
            })
        if k_shot == 10:
            log("\nStatistical significance (Wilcoxon, k=10):", log_file)
            for row in stat_rows_this_k:
                sig_marker = "**" if row["significant"] else "ns"
                log(f"  {row['comparison']}: p={row['p_value']:.4f} {sig_marker}, d={row['cohens_d']:.2f}", log_file)

        statistical_rows.extend(stat_rows_this_k)

    # 4. Full fine-tune (once)
    full_finetune_acc = None
    if run_full_finetune_flag:
        log("\nRunning Full Fine-Tune (uses all train+val data)...", log_file)
        ff_metrics = full_finetune.run_full_finetune(
            train_files, train_labels, val_files, val_labels, test_files, test_labels,
            n_classes=config.N_CLASSES, device=config.DEVICE, verbose=config.VERBOSE,
        )
        full_finetune_acc = ff_metrics["accuracy"]
        log(f"\nFull Fine-Tune (upper bound): {full_finetune_acc*100:.2f}%", log_file)

        for cname, stats in ff_metrics["per_class"].items():
            per_class_rows.append({
                "method": "full_finetune", "k_shot": "N/A", "class": cname,
                "precision": stats["precision"], "recall": stats["recall"],
                "f1": stats["f1"], "support": stats["support"],
            })
        visualization.plot_confusion_matrix(ff_metrics["confusion_matrix"], "full_finetune", "final")

    # 5. Save CSVs
    results_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    statistical_df = pd.DataFrame(statistical_rows)

    results_df.to_csv(os.path.join(results_dir, "results_summary.csv"), index=False, encoding="utf-8")
    per_class_df.to_csv(os.path.join(results_dir, "per_class_metrics.csv"), index=False, encoding="utf-8")
    statistical_df.to_csv(os.path.join(results_dir, "statistical_tests.csv"), index=False, encoding="utf-8")

    # 6. Additional visualizations
    visualization.plot_accuracy_vs_kshot(results_df, few_shot_methods, full_finetune_acc)
    visualization.plot_per_class_f1(per_class_df, k_shot=10 if 10 in k_shots else k_shots[0], class_names=config.CLASS_NAMES)

    if metric_weights_at_k10 is not None:
        visualization.plot_metric_weights_distribution(metric_weights_at_k10, k_shot=10)

    # t-SNE on test-set features (using the frozen extractor)
    if len(test_files) > 0:
        test_images = data_loader.load_images(test_files, config.IMAGE_SIZE)
        test_feats = extractor.extract_batch(test_images)
        visualization.plot_tsne_features(test_feats, test_labels, config.CLASS_NAMES)

    # 7. Summary table
    log("\n" + "=" * 60, log_file)
    log("SUMMARY (Mean +/- Std over runs)", log_file)
    log("=" * 60, log_file)
    pivot_mean = results_df.groupby(["method", "k_shot"])["accuracy"].mean().unstack() * 100
    pivot_std = results_df.groupby(["method", "k_shot"])["accuracy"].std().unstack() * 100
    for method in few_shot_methods:
        row_str = f"{METHOD_DISPLAY[method]:20s}"
        for k in k_shots:
            m = pivot_mean.loc[method, k] if k in pivot_mean.columns else float("nan")
            s = pivot_std.loc[method, k] if k in pivot_std.columns else float("nan")
            row_str += f"  {m:5.1f}+/-{s:4.1f}"
        log(row_str, log_file)
    if full_finetune_acc is not None:
        log(f"{'Full Fine-Tune':20s}  {full_finetune_acc*100:5.1f} (once)", log_file)
    log("=" * 60, log_file)

    log_file.close()
    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
