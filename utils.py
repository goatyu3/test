"""Utility functions for the cat vs. dog classifier project."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class AverageMeter:
    """Track the average and current value of a metric."""

    name: str
    value: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.value = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device: str | None = None) -> torch.device:
    """Return a torch.device, preferring CUDA if available."""

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy from raw logits."""

    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == targets).item()
    return correct / targets.size(0)


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: Iterable[str],
    normalize: bool = True,
    cmap: str = "Blues",
) -> plt.Figure:
    """Create a matplotlib figure containing a confusion matrix."""

    if normalize:
        confusion = confusion.astype(np.float64)
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion = np.divide(
            confusion,
            np.where(row_sums == 0, 1.0, row_sums),
            out=np.zeros_like(confusion, dtype=np.float64),
            where=row_sums != 0,
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(confusion, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = confusion.max() / 2.0 if confusion.size else 0.0
    for i, j in np.ndindex(confusion.shape):
        ax.text(
            j,
            i,
            f"{confusion[i, j]:.2f}" if normalize else f"{int(confusion[i, j])}",
            ha="center",
            va="center",
            color="white" if confusion[i, j] > thresh else "black",
        )

    fig.tight_layout()
    return fig


def save_confusion_matrix(
    confusion: np.ndarray,
    class_names: Iterable[str],
    output_path: Path | str,
    normalize: bool = True,
) -> Path:
    """Persist a confusion matrix plot to disk and return the path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_confusion_matrix(confusion, class_names, normalize=normalize)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_checkpoint(checkpoint_path: Path | str, map_location: str | torch.device | None = None) -> Dict:
    """Load a PyTorch checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=map_location)


def ensure_exists(path: Path | str) -> Path:
    """Create a directory path if it does not exist."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
