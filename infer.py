"""Inference script for the cat vs. dog classifier."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from PIL import Image

from datasets import get_transforms
from models import create_model
from utils import load_checkpoint, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on an image using a trained cat vs. dog classifier.")
    parser.add_argument("image", type=str, help="Path to the image file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/best.pt",
        help="Path to the trained model checkpoint (best.pt).",
    )
    parser.add_argument("--device", type=str, default=None, help="Computation device (cpu, cuda, cuda:0, ...).")
    parser.add_argument("--image-size", type=int, default=224, help="Image size used during training.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of top predictions to display.")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, Dict[int, str]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

    model = create_model(num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, idx_to_class


def preprocess_image(image_path: Path, image_size: int) -> torch.Tensor:
    transform = get_transforms(image_size=image_size, augment=False)[1]
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    checkpoint_path = Path(args.checkpoint)
    model, idx_to_class = load_model(checkpoint_path, device)

    inputs = preprocess_image(image_path, args.image_size).to(device)

    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_k = min(args.top_k, probabilities.numel())
    top_probs, top_indices = torch.topk(probabilities, top_k)

    print(f"Predictions for {image_path}:")
    for prob, idx in zip(top_probs, top_indices):
        class_name = idx_to_class[idx.item()]
        print(f"  {class_name}: {prob.item():.4f}")


if __name__ == "__main__":
    main()
