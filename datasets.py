"""Dataset utilities for the cat vs. dog classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224, augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return torchvision transforms for training and validation."""

    train_transforms: List[transforms.Compose] = [transforms.Resize((image_size, image_size))]
    if augment:
        train_transforms = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ]
    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return transforms.Compose(train_transforms), val_transforms


def _subset_dataset(dataset: datasets.ImageFolder, indices: Iterable[int]) -> Subset:
    return Subset(dataset, list(indices))


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    image_size: int = 224,
    augment: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader], List[str], Dict[str, int]]:
    """Create training and validation dataloaders from an ImageFolder dataset."""

    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    train_transform, val_transform = get_transforms(image_size=image_size, augment=augment)
    base_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    class_names = base_train_dataset.classes
    class_to_idx = base_train_dataset.class_to_idx

    if val_dir.exists() and any(val_dir.iterdir()):
        val_dataset: Optional[Dataset] = datasets.ImageFolder(val_dir, transform=val_transform)
        train_dataset = base_train_dataset
    else:
        if val_split <= 0 or val_split >= 1:
            raise ValueError("val_split must be between 0 and 1 when no validation directory is provided.")
        generator = torch.Generator().manual_seed(seed)
        num_samples = len(base_train_dataset)
        num_val = max(1, int(num_samples * val_split))
        train_dataset, val_indices = _split_datasets(base_train_dataset, num_val, generator)
        # val_indices is a Subset based on a dataset with train transforms. Create a separate dataset with validation transforms.
        base_val_dataset = datasets.ImageFolder(train_dir, transform=val_transform)
        val_dataset = _subset_dataset(base_val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    return train_loader, val_loader, class_names, class_to_idx


def _split_datasets(
    dataset: datasets.ImageFolder,
    num_val: int,
    generator: torch.Generator,
) -> Tuple[Subset, List[int]]:
    """Split a dataset into training subset and validation indices."""

    indices = torch.randperm(len(dataset), generator=generator)
    val_indices = indices[:num_val].tolist()
    train_indices = indices[num_val:].tolist()
    train_subset = _subset_dataset(dataset, train_indices)
    return train_subset, val_indices

