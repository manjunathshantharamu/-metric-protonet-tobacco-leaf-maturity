"""
feature_extractor.py — Frozen MobileNetV3-Small feature extractor (PyTorch).

PyTorch-only by design decision: the pipeline previously supported a NumPy
hand-crafted-feature fallback and a CLIP backend, but both were dropped
(CLIP gave flat, near-random ~20% accuracy due to domain gap; the NumPy
fallback added complexity without being part of the tested/reported pipeline).

Architecture:
    Image -> [Frozen MobileNetV3-Small backbone] -> AdaptiveAvgPool2d(1) ->
              flatten -> 576-dim feature vector

All backbone parameters are frozen (requires_grad=False); only downstream
methods (e.g. Metric ProtoNet's diagonal metric, transfer learning's linear
head) are trained on top of these fixed features. full_finetune.py is the one
exception, where it loads its own unfrozen copy of the backbone.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

import config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device(device_pref: str = "auto") -> torch.device:
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


class FeatureExtractor:
    """Frozen MobileNetV3-Small feature extractor.

    extract_batch(images) -> (N, 576) float32 NumPy array of pooled features.
    """

    def __init__(self, backbone: str = "mobilenet_v3_small", pretrained: bool = True,
                 device: str = "auto"):
        if backbone != "mobilenet_v3_small":
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.device = get_device(device)

        weights = None
        if pretrained:
            try:
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = None  # older torchvision: fall back to pretrained=True below

        if weights is not None:
            backbone_model = models.mobilenet_v3_small(weights=weights)
        else:
            backbone_model = models.mobilenet_v3_small(pretrained=pretrained)

        # Keep only the convolutional feature backbone (drop the classifier head).
        self.backbone = backbone_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.to(self.device)
        self.pool.to(self.device)

        self._preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(config.IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._feature_dim = self._infer_feature_dim()

    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], device=self.device)
            feat = self.pool(self.backbone(dummy))
            return int(feat.view(1, -1).shape[1])

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @torch.no_grad()
    def extract_batch(self, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Extract 576-dim features for a batch of images.

        Parameters
        ----------
        images : np.ndarray, shape (N, H, W, 3), dtype uint8
        batch_size : int
            Mini-batch size for the forward pass (memory control only; does
            not affect the result).

        Returns
        -------
        np.ndarray, shape (N, feature_dim), dtype float32
        """
        n = len(images)
        out = np.zeros((n, self.feature_dim), dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = images[start:end]
            tensors = torch.stack([self._preprocess(img) for img in batch]).to(self.device)
            feats = self.pool(self.backbone(tensors))
            feats = feats.view(feats.size(0), -1)
            out[start:end] = feats.cpu().numpy().astype(np.float32)

        return out

    def preprocess_single(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single (H, W, 3) uint8 image into a normalized tensor
        (used by methods that need raw tensors, e.g. full_finetune.py)."""
        return self._preprocess(image)
