"""PyTorch CNN classifier for wafer map defect inspection.

Small CNN: two conv blocks (conv -> relu -> maxpool), global flatten,
two dense layers, softmax output. Class-weighted cross-entropy handles the
imbalanced 9-class taxonomy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class WaferCNN(nn.Module):
    def __init__(self, n_classes: int = 9, in_ch: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def embed(self, x: np.ndarray) -> np.ndarray:
        """Penultimate-layer embeddings (for k-means discovery)."""
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).unsqueeze(1)
            feats = self.features(t)
            flat = torch.flatten(feats, 1)
            # Second-to-last layer of the classifier: Linear(8192->256) + ReLU.
            h = self.classifier[1](flat)
            h = self.classifier[2](h)
        return h.cpu().numpy()


def make_model(n_classes: int = 9, seed: int = 0) -> WaferCNN:
    torch.manual_seed(seed)
    return WaferCNN(n_classes=n_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
