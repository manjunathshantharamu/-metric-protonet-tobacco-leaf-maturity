"""
methods/full_finetune.py — Method 4: Full fine-tune (upper bound).

Architecture:
    Image -> [CNN (ALL layers trainable)] -> [Classifier (trainable)] -> Softmax

Procedure:
    1. Uses ALL available data (not few-shot — full train+val set).
    2. Fine-tunes the entire MobileNetV3-Small backbone + a new classifier
       head, with a smaller LR for the pretrained backbone and a larger LR
       for the freshly-initialized head.
    3. Data augmentation applied (flips, rotation, color jitter).
    4. Early stopping on validation accuracy.
    5. Evaluated once on the held-out test set — NOT per k-shot, NOT per run.

This provides the upper-bound reference point: what accuracy is achievable
with unlimited (well, all available) labeled data, against which the
few-shot methods' data-efficiency can be judged.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import config
import evaluation


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class _ImageDataset(Dataset):
    def __init__(self, filepaths: List[str], labels: np.ndarray, transform):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        with Image.open(self.filepaths[idx]) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, int(self.labels[idx])


def _build_transforms():
    aug = config.AUGMENTATION
    jitter = aug.get("color_jitter", 0.0)

    train_transform = T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.RandomHorizontalFlip() if aug.get("horizontal_flip") else T.Lambda(lambda x: x),
        T.RandomVerticalFlip() if aug.get("vertical_flip") else T.Lambda(lambda x: x),
        T.RandomRotation(aug.get("rotation", 0)) if aug.get("rotation") else T.Lambda(lambda x: x),
        T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=min(jitter, 0.5))
            if jitter else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def _build_model(n_classes: int) -> nn.Module:
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    except AttributeError:
        model = models.mobilenet_v3_small(pretrained=True)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    return model


def run_full_finetune(
    train_files: List[str],
    train_labels: np.ndarray,
    val_files: List[str],
    val_labels: np.ndarray,
    test_files: List[str],
    test_labels: np.ndarray,
    n_classes: int = 3,
    device: str = "auto",
    verbose: int = 1,
) -> Dict:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    torch.manual_seed(config.RANDOM_SEED)

    train_transform, eval_transform = _build_transforms()

    train_ds = _ImageDataset(train_files, train_labels, train_transform)
    val_ds = _ImageDataset(val_files, val_labels, eval_transform)
    test_ds = _ImageDataset(test_files, test_labels, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=config.TL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.TL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.TL_BATCH_SIZE, shuffle=False)

    model = _build_model(n_classes).to(device)

    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("classifier")]

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": config.TL_LR_FINETUNE},
        {"params": head_params, "lr": config.TL_LR},
    ])

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.TL_EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(total, 1)

        if verbose >= 2:
            print(f"    [full_finetune] epoch {epoch+1}/{config.TL_EPOCHS}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.TL_PATIENCE:
                if verbose >= 1:
                    print(f"    [full_finetune] early stopping at epoch {epoch+1} (best val_acc={best_val_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on the held-out test set.
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_probs.append(probs)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    metrics = evaluation.compute_metrics(
        all_labels, all_preds, y_proba=all_probs, n_classes=n_classes
    )
    metrics["predictions"] = all_preds
    metrics["probabilities"] = all_probs
    metrics["best_val_accuracy"] = best_val_acc

    return metrics
