"""
data_loader.py — Dataset scanning, stratified splitting, and few-shot episode sampling.

Responsibilities
-----------------
1. Scan the dataset folder -> build an index of (filepath, class_name, class_index).
2. Stratified train/val/test split (70/15/15 of the whole set, i.e. 80/20 then
   the 80% train portion split 85/15 into train/val).
3. Few-shot episode sampling: for a given k-shot value and run index, sample a
   support set (k images per class) from TRAIN, and use ALL remaining TRAIN
   images as the query set.
4. Image loading (PIL -> resize -> RGB -> uint8 NumPy array).
5. Data augmentation for training-based methods (transfer_learning, full_finetune).
"""

import os
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import config


def load_dataset(root_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Scan the dataset folder and build the image index.

    Expected layout:
        root_path/
            Unripe/   *.jpg / *.jpeg / *.png
            Ripe/     ...
            Overripe/ ...

    Returns
    -------
    filepaths : List[str]
        Absolute paths to every image found.
    labels : np.ndarray[str]
        Class name for each image (aligned with filepaths).
    class_indices : np.ndarray[int]
        Integer class index for each image, per config.CLASS_NAMES ordering.
    """
    if not os.path.isdir(root_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {root_path}\n"
            f"Set config.DATASET_PATH, the TOBACCO_DATASET_PATH environment "
            f"variable, or pass --dataset to point at the correct folder."
        )

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    filepaths: List[str] = []
    labels: List[str] = []
    class_indices: List[int] = []

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(root_path, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected class folder not found: {class_dir}\n"
                f"CLASS_NAMES in config.py must match the dataset's subfolder names."
            )
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(valid_ext):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(class_name)
                class_indices.append(class_idx)

    if len(filepaths) == 0:
        raise RuntimeError(f"No images found under {root_path}")

    return filepaths, np.array(labels), np.array(class_indices, dtype=np.int64)


def stratified_split(
    filepaths: List[str],
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """Stratified train/val/test split.

    First splits off `test_size` fraction as the test set, then splits the
    remaining "train" portion again by `val_size` fraction into train/val,
    both stratified by class label.

    Returns
    -------
    dict with keys "train", "val", "test", each mapping to (filepaths, labels).
    """
    filepaths = np.array(filepaths)

    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        filepaths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=seed,
    )

    return {
        "train": (list(train_files), train_labels),
        "val": (list(val_files), val_labels),
        "test": (list(test_files), test_labels),
    }


def sample_episode(
    train_files: List[str],
    train_labels: np.ndarray,
    k_shot: int,
    n_classes: int,
    seed: int,
    class_names: List[str] = None,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """Sample a few-shot episode from the TRAIN split.

    Support set: k images per class (random, without replacement).
    Query set: ALL remaining TRAIN images (i.e. every train image not chosen
    for the support set), not a further subsample.

    A distinct seed must be passed per (run, k_shot) combination by the caller,
    per config's reproducibility convention: seed = RANDOM_SEED + run_idx*100 + k_shot.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    rng = np.random.RandomState(seed)
    train_files = np.array(train_files)

    support_idx = []
    for class_idx in range(n_classes):
        class_positions = np.where(train_labels == class_idx)[0]
        if len(class_positions) < k_shot:
            raise ValueError(
                f"Class '{class_names[class_idx]}' has only {len(class_positions)} "
                f"training images, fewer than k_shot={k_shot}."
            )
        chosen = rng.choice(class_positions, size=k_shot, replace=False)
        support_idx.extend(chosen.tolist())

    support_idx = np.array(sorted(support_idx))
    all_idx = np.arange(len(train_files))
    query_idx = np.setdiff1d(all_idx, support_idx)

    support_files = train_files[support_idx].tolist()
    support_labels = train_labels[support_idx]
    query_files = train_files[query_idx].tolist()
    query_labels = train_labels[query_idx]

    return support_files, support_labels, query_files, query_labels


def load_images(filepaths: List[str], image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and resize a list of image files.

    Returns an (N, H, W, 3) uint8 NumPy array in RGB order.
    """
    images = np.zeros((len(filepaths), image_size[0], image_size[1], 3), dtype=np.uint8)
    for i, fp in enumerate(filepaths):
        with Image.open(fp) as img:
            img = img.convert("RGB").resize(image_size, Image.BILINEAR)
            images[i] = np.array(img, dtype=np.uint8)
    return images


def augment_images(images: np.ndarray, aug_config: dict = None) -> np.ndarray:
    """Apply data augmentation to a batch of uint8 (N, H, W, 3) images.

    Used only by training-based methods (transfer_learning's optional PyTorch
    path, and full_finetune). Implemented with PIL for simplicity/determinism
    outside of the torch DataLoader path; the torch-based training loops use
    torchvision transforms directly (see methods/full_finetune.py) — this
    function is provided for any ad-hoc augmentation needs (e.g. within
    evaluation scripts or notebooks).
    """
    if aug_config is None:
        aug_config = config.AUGMENTATION

    rng = np.random.RandomState(config.RANDOM_SEED)
    out = images.copy()

    for i in range(len(out)):
        img = Image.fromarray(out[i])

        if aug_config.get("horizontal_flip") and rng.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if aug_config.get("vertical_flip") and rng.rand() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if aug_config.get("rotation"):
            angle = rng.uniform(-aug_config["rotation"], aug_config["rotation"])
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        if aug_config.get("color_jitter"):
            from PIL import ImageEnhance
            jitter = aug_config["color_jitter"]
            for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                factor = 1.0 + rng.uniform(-jitter, jitter)
                img = enhancer_cls(img).enhance(max(0.0, factor))

        out[i] = np.array(img, dtype=np.uint8)

    return out
