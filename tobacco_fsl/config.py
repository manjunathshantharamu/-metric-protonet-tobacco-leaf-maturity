"""
config.py — Central configuration for the Tobacco Leaf Few-Shot Classification pipeline.

All tunable parameters live here so that every module (data loading, feature
extraction, methods, evaluation, visualization, and the CLI runners) reads from
a single source of truth. This keeps experiments reproducible: the same
config.py + the same dataset + the same RANDOM_SEED must always produce the
same results/ outputs.
"""

import os

# ---------------------------------------------------------------------------
# 1. DATASET
# ---------------------------------------------------------------------------
# Dataset path is configurable via (in priority order):
#   1. --dataset CLI flag (see run_experiment.py / run_ablation.py)
#   2. TOBACCO_DATASET_PATH environment variable
#   3. The default below
# This avoids hard-coding a machine-specific path into the pipeline.
DATASET_PATH = os.environ.get(
    "TOBACCO_DATASET_PATH",
    r"D:\Research\MalliWork\Tobacco\Dataset\Tobacco",
)

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Unripe", "Ripe", "Overripe"]
N_CLASSES = 3
RANDOM_SEED = 42
TEST_SIZE = 0.2  # 80/20 train/test split, then train split into train/val (85/15)
VAL_SIZE = 0.15  # fraction of the 80% "train" portion held out for validation

# ---------------------------------------------------------------------------
# 2. FEW-SHOT SETTINGS
# ---------------------------------------------------------------------------
K_SHOTS = [5, 10, 15, 20, 30]
N_RUNS = 10  # independent runs per (method, k-shot)

# ---------------------------------------------------------------------------
# 3. MODEL / BACKBONE
# ---------------------------------------------------------------------------
BACKBONE = "mobilenet_v3_small"
FEATURE_DIM = 576  # MobileNetV3-Small's pooled feature dimension (confirmed)
PRETRAINED = True

# ---------------------------------------------------------------------------
# 4. METRIC PROTONET HYPERPARAMETERS (PROPOSED METHOD)
# ---------------------------------------------------------------------------
METRIC_PROTONET_EPOCHS = 200
METRIC_PROTONET_LR = 0.01
METRIC_PROTONET_WEIGHT_DECAY = 1.0  # L2 reg; ablation A3 found lambda=1.0 optimal at k>=10
METRIC_PROTONET_TEMPERATURE = 10.0

# ---------------------------------------------------------------------------
# 5. TRANSFER LEARNING / FULL FINE-TUNE
# ---------------------------------------------------------------------------
TL_EPOCHS = 50
TL_LR = 1e-3
TL_LR_FINETUNE = 1e-4  # backbone LR during full fine-tune
TL_BATCH_SIZE = 32
TL_PATIENCE = 10  # early stopping patience (epochs without val improvement)

# ---------------------------------------------------------------------------
# 6. PROTOTYPICAL NETWORK (STANDARD)
# ---------------------------------------------------------------------------
PROTONET_TEMPERATURE = 10.0

# ---------------------------------------------------------------------------
# 7. OUTPUT
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ---------------------------------------------------------------------------
# 8. DEVICE
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("DEVICE", "auto")  # "auto", "cpu", "cuda"

# ---------------------------------------------------------------------------
# 9. AUGMENTATION (training methods only: transfer_learning, full_finetune)
# ---------------------------------------------------------------------------
AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotation": 30,       # degrees
    "color_jitter": 0.2,  # brightness/contrast/saturation/hue +/- this value
    "random_crop": True,
}

# ---------------------------------------------------------------------------
# 10. METHODS TO RUN (order matters for output tables/plots)
# ---------------------------------------------------------------------------
# NOTE: CLIP zero-shot and the legacy color-attention ProtoNet were dropped
# from this pipeline (see project history) because CLIP produced flat,
# near-random results (~20%) due to domain gap, and color_protonet was an
# earlier, weaker version of metric_protonet.
METHODS = [
    "transfer_learning",
    "protonet",
    "metric_protonet",
    "full_finetune",
]

# ---------------------------------------------------------------------------
# 11. LOGGING
# ---------------------------------------------------------------------------
VERBOSE = 2  # 0=silent, 1=progress, 2=detailed
