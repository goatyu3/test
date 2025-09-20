"""Model definitions for the cat vs. dog classifier."""
from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torchvision import models


def create_model(num_classes: int, pretrained: bool = True, dropout: Optional[float] = 0.0) -> nn.Module:
    """Return a ResNet18 model adapted for the required number of classes."""

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features
    classifier_layers = []
    if dropout and dropout > 0:
        classifier_layers.append(nn.Dropout(p=dropout))
    classifier_layers.append(nn.Linear(in_features, num_classes))
    model.fc = nn.Sequential(*classifier_layers) if len(classifier_layers) > 1 else classifier_layers[0]
    return model
