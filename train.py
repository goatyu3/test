"""Training script for the cat vs. dog classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import create_dataloaders
from models import create_model
from utils import AverageMeter, accuracy_from_logits, ensure_exists, save_confusion_matrix, select_device, set_seed


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    loss_meter = AverageMeter(name="train_loss")
    acc_meter = AverageMeter(name="train_acc")

    progress = tqdm(dataloader, desc="Train", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc = accuracy_from_logits(outputs, targets)
        loss_meter.update(loss.item(), n=inputs.size(0))
        acc_meter.update(acc, n=inputs.size(0))
        progress.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader | None,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    if dataloader is None:
        return float("nan"), float("nan"), np.array([]), np.array([])

    model.eval()
    loss_meter = AverageMeter(name="val_loss")
    acc_meter = AverageMeter(name="val_acc")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Validate", leave=False)
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc = accuracy_from_logits(outputs, targets)
            loss_meter.update(loss.item(), n=inputs.size(0))
            acc_meter.update(acc, n=inputs.size(0))
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_preds) if all_preds else np.array([])
    targets = np.concatenate(all_targets) if all_targets else np.array([])
    return loss_meter.avg, acc_meter.avg, predictions, targets


def save_best_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    class_to_idx: Dict[str, int],
    args: argparse.Namespace,
) -> Path:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "class_to_idx": class_to_idx,
        "args": vars(args),
    }
    checkpoint_path = output_dir / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cat vs. dog classifier using PyTorch.")
    parser.add_argument("data_dir", type=str, help="Path to dataset root (expects train/ and optionally val/ directories).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Optional dropout applied to the classifier head.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet pre-trained weights.")
    parser.add_argument("--image-size", type=int, default=224, help="Image size used for transforms.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader worker processes.")
    parser.add_argument("--device", type=str, default=None, help="Computation device (cpu, cuda, cuda:0, ...).")
    parser.add_argument("--output-dir", type=str, default="runs", help="Directory to store checkpoints and logs.")
    parser.add_argument("--split", type=float, default=0.2, help="Validation split ratio if no val/ directory is present.")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = ensure_exists(args.output_dir)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    train_loader, val_loader, class_names, class_to_idx = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.split,
        image_size=args.image_size,
        augment=not args.no_augment,
        seed=args.seed,
    )

    model = create_model(num_classes=len(class_names), pretrained=not args.no_pretrained, dropout=args.dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = float("-inf")
    best_epoch = 0
    best_predictions = np.array([])
    best_targets = np.array([])

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, predictions, targets = evaluate(model, val_loader, criterion, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        if val_loader is not None:
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

        if val_loader is not None and val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_predictions = predictions
            best_targets = targets
            save_path = save_best_checkpoint(output_dir, model, optimizer, epoch, class_to_idx, args)
            print(f"New best model saved to {save_path} (val_acc={best_acc:.4f})")

    if val_loader is not None and best_predictions.size and best_targets.size:
        cm = confusion_matrix(best_targets, best_predictions)
        cm_path = save_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png", normalize=True)
        writer.add_image(
            "confusion_matrix",
            plt_to_tensor(cm_path),
            global_step=best_epoch,
        )
        print(f"Confusion matrix saved to {cm_path}")
    else:
        print("Validation data unavailable; skipping confusion matrix generation.")

    if val_loader is None:
        save_path = save_best_checkpoint(output_dir, model, optimizer, args.epochs, class_to_idx, args)
        print(f"Final model saved to {save_path} (no validation set available)")
        best_epoch = args.epochs
        best_acc_value = None
    else:
        best_acc_value = best_acc

    metadata_path = output_dir / "training_summary.json"
    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_acc_value,
        "class_names": class_names,
        "output_dir": str(output_dir),
        "device": str(device),
    }
    metadata_path.write_text(json.dumps(summary, indent=2))
    print(f"Training summary written to {metadata_path}")

    writer.close()


def plt_to_tensor(image_path: Path | str) -> np.ndarray:
    """Load an image file and return a TensorBoard-compatible numpy array."""

    import matplotlib.image as mpimg

    img = mpimg.imread(image_path)
    if img.ndim == 2:  # grayscale
        img = np.stack([img] * 3, axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))
    return img.astype(np.float32)


if __name__ == "__main__":
    main()
