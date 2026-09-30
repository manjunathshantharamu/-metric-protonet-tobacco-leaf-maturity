"""
run_complexity_analysis.py — Computational complexity analysis.

Measures, per method:
    1. Trainable parameter count
    2. FLOPs for a single inference (forward pass)
    3. Inference time on CPU (and GPU, if available)
    4. Model size estimate (MB, float32)

Uses a synthetic input (torch.randn(1, 3, 224, 224)) — no dataset required.

Usage:
    python run_complexity_analysis.py
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def _build_backbone():
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    except AttributeError:
        model = models.mobilenet_v3_small(pretrained=True)
    return model


def _count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def _measure_flops(model, input_tensor):
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return float(flops)
    except Exception:
        return float("nan")


def _measure_inference_time(model, input_tensor, n_reps=100):
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(input_tensor)

        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            model(input_tensor)
            times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    return float(times.mean()), float(times.std())


def main():
    results_dir = config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cpu")  # CPU by default; extend to CUDA below if available
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], device=device)

    backbone_full = _build_backbone().to(device)
    backbone_full.eval()

    feature_backbone = backbone_full.features
    pool = nn.AdaptiveAvgPool2d(1)

    class _FeatureNet(nn.Module):
        def __init__(self, backbone, pool):
            super().__init__()
            self.backbone = backbone
            self.pool = pool

        def forward(self, x):
            return self.pool(self.backbone(x)).flatten(1)

    feature_net = _FeatureNet(feature_backbone, pool).to(device)
    feature_dim = config.FEATURE_DIM

    backbone_flops = _measure_flops(feature_net, dummy_input)
    backbone_cpu_mean, backbone_cpu_std = _measure_inference_time(feature_net, dummy_input)

    rows = []

    # Transfer Learning: Linear(D, 3)
    tl_params = feature_dim * config.N_CLASSES + config.N_CLASSES
    rows.append({
        "method": "transfer_learning",
        "trainable_params": tl_params,
        "frozen_params": _count_params(feature_backbone),
        "flops": backbone_flops,
        "inference_time_cpu_ms_mean": backbone_cpu_mean,
        "inference_time_cpu_ms_std": backbone_cpu_std,
        "model_size_mb": (tl_params * 4) / 1e6,
    })

    # ProtoNet: no trainable params
    rows.append({
        "method": "protonet",
        "trainable_params": 0,
        "frozen_params": _count_params(feature_backbone),
        "flops": backbone_flops,
        "inference_time_cpu_ms_mean": backbone_cpu_mean,
        "inference_time_cpu_ms_std": backbone_cpu_std,
        "model_size_mb": 0.0,
    })

    # Metric ProtoNet: D trainable params (one per feature dimension)
    metric_flops = backbone_flops + feature_dim * config.N_CLASSES * 3  # rough extra cost of weighted distance
    rows.append({
        "method": "metric_protonet",
        "trainable_params": feature_dim,
        "frozen_params": _count_params(feature_backbone),
        "flops": metric_flops,
        "inference_time_cpu_ms_mean": backbone_cpu_mean,  # dominated by backbone; metric layer is negligible
        "inference_time_cpu_ms_std": backbone_cpu_std,
        "model_size_mb": (feature_dim * 4) / 1e6,
    })

    # Full Fine-Tune: all MobileNetV3-Small params trainable
    full_model = _build_backbone().to(device)
    in_features = full_model.classifier[-1].in_features
    full_model.classifier[-1] = nn.Linear(in_features, config.N_CLASSES)
    full_model.eval()

    full_flops = _measure_flops(full_model, dummy_input)
    full_cpu_mean, full_cpu_std = _measure_inference_time(full_model, dummy_input)
    full_params = _count_params(full_model)

    rows.append({
        "method": "full_finetune",
        "trainable_params": full_params,
        "frozen_params": 0,
        "flops": full_flops,
        "inference_time_cpu_ms_mean": full_cpu_mean,
        "inference_time_cpu_ms_std": full_cpu_std,
        "model_size_mb": (full_params * 4) / 1e6,
    })

    # GPU timings if available
    if torch.cuda.is_available():
        device_gpu = torch.device("cuda")
        dummy_gpu = dummy_input.to(device_gpu)
        feature_net_gpu = _FeatureNet(feature_backbone, pool).to(device_gpu)
        gpu_mean, gpu_std = _measure_inference_time(feature_net_gpu, dummy_gpu)
        for row in rows[:3]:
            row["inference_time_gpu_ms_mean"] = gpu_mean
            row["inference_time_gpu_ms_std"] = gpu_std

        full_model_gpu = full_model.to(device_gpu)
        full_gpu_mean, full_gpu_std = _measure_inference_time(full_model_gpu, dummy_gpu)
        rows[3]["inference_time_gpu_ms_mean"] = full_gpu_mean
        rows[3]["inference_time_gpu_ms_std"] = full_gpu_std
    else:
        for row in rows:
            row["inference_time_gpu_ms_mean"] = float("nan")
            row["inference_time_gpu_ms_std"] = float("nan")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "complexity_analysis.csv"), index=False, encoding="utf-8")

    # Plots
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df["method"], df["trainable_params"], color="#2ca02c")
    ax.set_ylabel("Trainable parameters")
    ax.set_title("Trainable Parameter Count by Method")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "complexity_params.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df["method"], df["inference_time_cpu_ms_mean"], yerr=df["inference_time_cpu_ms_std"], color="#1f77b4")
    ax.set_ylabel("Inference time (ms, CPU)")
    ax.set_title("Inference Time by Method (CPU, single image)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "complexity_inference_time.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df["method"], df["model_size_mb"], color="#9467bd")
    ax.set_ylabel("Model size (MB, trainable params only)")
    ax.set_title("Trainable Model Size by Method")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "complexity_model_size.png"))
    plt.close(fig)

    print(df.to_string(index=False))
    print(f"\nComplexity analysis saved to: {results_dir}")


if __name__ == "__main__":
    main()
