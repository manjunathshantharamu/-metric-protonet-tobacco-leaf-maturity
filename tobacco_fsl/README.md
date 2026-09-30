# Tobacco Leaf Few-Shot Classification

A few-shot learning research pipeline for grading tobacco leaf ripeness
(**Unripe**, **Ripe**, **Overripe**) from phone-camera-style images, targeting
mobile/edge deployment for farmers and low-cost robotic grading arms.

## Why few-shot learning

Real-world deployment sees variety in tobacco varieties, lighting, and field
conditions. Few-shot learning enables rapid adaptation to new conditions with
minimal labeled data, using a lightweight frozen backbone (MobileNetV3-Small)
suitable for mobile/edge inference.

## Methods compared

| Method | File | Trainable params | Notes |
|---|---|---|---|
| Transfer Learning | `methods/transfer_learning.py` | ~1.7K | Frozen CNN + logistic regression head |
| ProtoNet | `methods/protonet.py` | 0 | Standard Prototypical Network (Snell et al. 2017), inference only |
| **Metric-ProtoNet (proposed)** | `methods/metric_protonet.py` | 576 | Learnable diagonal distance metric over frozen features |
| Full Fine-Tune | `methods/full_finetune.py` | ~2.5M | Upper bound — uses all data, not few-shot |

> CLIP zero-shot and an earlier color-attention ProtoNet variant were
> evaluated during development and dropped: CLIP produced flat, near-random
> (~20%) accuracy due to domain gap, and the color-attention variant was
> superseded by the simpler, better-performing Metric-ProtoNet.

## The proposed method: Metric-ProtoNet

Pretrained CNN features (576-dim) are over-complete for a narrow,
domain-specific few-shot task — they encode information irrelevant to leaf
ripeness (shape, texture, generic ImageNet semantics). Standard ProtoNet
weighs every dimension equally under Euclidean distance, letting irrelevant
dimensions degrade classification.

Metric-ProtoNet learns a diagonal metric `w ∈ R^576` (one scalar per feature
dimension) via gradient descent on the support set:

```
d(query, prototype) = sum_i  w_i * (q_i - p_i)^2
```

`w_i ≈ 1` marks a useful dimension; `w_i ≈ 0` suppresses an irrelevant one.
It is implemented as a `torch.nn.Parameter` trained with full PyTorch
autograd (Adam, lr=0.01, 200 epochs, no weight decay) — this is
mathematically equivalent to the direct+indirect closed-form gradient through
the prototype computation, without hand-derivation risk.

## Project layout

```
tobacco_fsl/
├── config.py                       # Central configuration (all parameters)
├── data_loader.py                  # Dataset loading, stratified split, episode sampling
├── feature_extractor.py            # Frozen MobileNetV3-Small (PyTorch)
├── evaluation.py                   # Metrics, confusion matrices, Wilcoxon tests
├── visualization.py                # Publication-ready plots (300 DPI)
├── run_experiment.py               # Main experiment runner (CLI)
├── run_ablation.py                 # Ablation study runner
├── run_complexity_analysis.py      # Computational complexity analysis
├── generate_results_report.py      # Excel report + additional graphs
├── requirements.txt
├── methods/
│   ├── __init__.py
│   ├── transfer_learning.py
│   ├── protonet.py
│   ├── metric_protonet.py
│   └── full_finetune.py
└── results/                        # Auto-created output directory
```

## Dataset

```
<DATASET_PATH>/
├── Unripe/    *.jpg / *.jpeg / *.png
├── Ripe/
└── Overripe/
```

`DATASET_PATH` is configurable (not hard-coded):

1. `--dataset /path/to/data` CLI flag, or
2. `TOBACCO_DATASET_PATH` environment variable, or
3. The default in `config.py`.

Split: stratified 70/15/15 train/val/test (`RANDOM_SEED=42`).

## Running the pipeline

```bash
pip install -r requirements.txt

# 1. Main experiment (CSVs + base plots)
python run_experiment.py

# 2. Ablation study (A1-A4)
python run_ablation.py

# 3. Computational complexity analysis (no dataset needed)
python run_complexity_analysis.py

# 4. Excel report + additional plots
python generate_results_report.py
```

Quick smoke test (1 run, k=10 only):

```bash
python run_experiment.py --quick
python run_ablation.py --quick
```

Custom dataset path or method subset:

```bash
python run_experiment.py --dataset /path/to/data --methods transfer_learning,protonet
```

## Reproducibility

- `RANDOM_SEED = 42` seeds NumPy and PyTorch at the relevant entry points.
- Each few-shot episode uses `seed = RANDOM_SEED + run_idx * 100 + k_shot`.
- CNN features are extracted once per episode and reused across all methods
  in that episode (not re-extracted per method).
- Support/query split: support = k images/class from TRAIN; query = ALL
  remaining TRAIN images. The held-out TEST split is used only by
  Full Fine-Tune's final evaluation.

## Outputs (`results/`)

CSV: `results_summary.csv`, `per_class_metrics.csv`, `statistical_tests.csv`,
`ablation_results.csv`, `ablation_summary.csv`, `complexity_analysis.csv`.

Plots (300 DPI): accuracy-vs-k-shot curves, per-method/per-k confusion
matrices, t-SNE of the feature space, per-class F1 bars, box plots of
run-to-run accuracy, the learned metric-weight histogram, ablation charts
(A1–A4), complexity bar charts, an accuracy heatmap, a per-class F1 radar
chart, an accuracy-gain-over-ProtoNet chart, statistical-significance
heatmaps, and a publication-ready summary table image.

Excel: `experiment_results.xlsx` (7 sheets — Summary, Raw_Results,
Pivot_Mean, Pivot_Std, Per_Class_Metrics, Statistical_Tests, Best_Per_K).
