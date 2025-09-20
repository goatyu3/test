"""Dataset utilities for the cat vs. dog classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


CAT_BREEDS = {
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
}

DOG_BREEDS = {
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "boxer",
    "chihuahua",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "miniature_pinscher",
    "newfoundland",
    "pomeranian",
    "pug",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier",
}

PET_BREED_TO_CATEGORY: Dict[str, str] = {
    **{breed: "cat" for breed in CAT_BREEDS},
    **{breed: "dog" for breed in DOG_BREEDS},
}


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


def _subset_dataset(dataset: Dataset, indices: Iterable[int]) -> Subset:
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
    """Create training and validation dataloaders for the Oxford-IIIT Pet dataset.

    Images are downloaded automatically (if needed) to ``data_dir`` using
    :class:`torchvision.datasets.OxfordIIITPet`. Only the coarse ``cat`` and ``dog``
    labels are exposed to the training loop. When ``val_split`` is greater than 0,
    a portion of the training samples is reserved for validation.
    """

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_transform, val_transform = get_transforms(image_size=image_size, augment=augment)

    full_train_dataset = datasets.OxfordIIITPet(
        root=str(data_path),
        split="trainval",
        target_types="category",
        transform=train_transform,
        download=True,
    )

    class_names = ["cat", "dog"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def _make_target_transform(classes: Iterable[str]):
        class_list = list(classes)

        def _target_transform(target: int) -> int:
            class_name = class_list[target]
            try:
                category = PET_BREED_TO_CATEGORY[class_name]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown Oxford-IIIT Pet class '{class_name}' encountered while remapping"
                ) from exc
            return class_to_idx[category]

        return _target_transform

    full_train_dataset.target_transform = _make_target_transform(full_train_dataset.classes)

    if val_split < 0:
        raise ValueError("val_split must be non-negative.")

    val_dataset: Optional[Dataset] = None
    if val_split > 0:
        if val_split >= 1:
            raise ValueError("val_split must be between 0 and 1.")
        generator = torch.Generator().manual_seed(seed)
        num_samples = len(full_train_dataset)
        num_val = max(1, int(num_samples * val_split))
        train_indices, val_indices = _split_indices(num_samples, num_val, generator)
        train_dataset = _subset_dataset(full_train_dataset, train_indices)

        val_base_dataset = datasets.OxfordIIITPet(
            root=str(data_path),
            split="trainval",
            target_types="category",
            transform=val_transform,
            download=False,
        )
        val_base_dataset.target_transform = _make_target_transform(val_base_dataset.classes)
        val_dataset = _subset_dataset(val_base_dataset, val_indices)
    else:
        train_dataset = full_train_dataset

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


def _split_indices(
    num_samples: int,
    num_val: int,
    generator: torch.Generator,
) -> Tuple[List[int], List[int]]:
    """Split indices for training and validation subsets."""

    indices = torch.randperm(num_samples, generator=generator)
    val_indices = indices[:num_val].tolist()
    train_indices = indices[num_val:].tolist()
    return train_indices, val_indices

