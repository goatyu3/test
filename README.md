# Cat vs. Dog Classifier (PyTorch)

This project trains a PyTorch image classifier to distinguish between cats and dogs. The
pipeline uses `torchvision.datasets.ImageFolder` so you can quickly plug in your own dataset
with the following structure:

```
<dataset_root>/
├── train/
│   ├── cats/
│   └── dogs/
└── val/ (optional)
    ├── cats/
    └── dogs/
```

If you do not provide a `val/` directory you can let the training script split the training set
on the fly via `--split` (default `0.2`).

## Installation (Windows PowerShell)

```powershell
# Clone repository and enter the project folder
cd path\to\your\workspace
git clone https://github.com/goatyu3/test.git
cd test

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Training

Run training by pointing to the dataset root. The best checkpoint is always saved to
`runs/best.pt`, TensorBoard logs are stored in `runs/tensorboard`, and the normalized
confusion matrix is exported as `runs/confusion_matrix.png`.

```powershell
# Train using an explicit validation set
python train.py D:\\data\\cats_dogs --epochs 15 --batch-size 32

# Train by splitting the training folder 80/20 (no val/ folder required)
python train.py D:\\data\\cats_dogs --epochs 15 --split 0.2
```

During training you can monitor metrics with TensorBoard:

```powershell
tensorboard --logdir runs/tensorboard
```

A `training_summary.json` file summarises the best epoch, accuracy and metadata for
future reference.

## Inference

After training, run inference against a single image using the saved checkpoint:

```powershell
python infer.py D:\\data\\sample.jpg --checkpoint runs\\best.pt
```

The script prints the top predictions with their probabilities. Use `--device cuda`
(if available) to run inference on a GPU.

## Project Files

- `train.py` – training loop with TensorBoard logging, checkpointing, and confusion-matrix export.
- `infer.py` – loads `runs/best.pt` and predicts the class of a single image.
- `models.py` – model factory (ResNet18 backbone with optional dropout).
- `datasets.py` – ImageFolder data loading utilities with optional train/validation split.
- `utils.py` – helper utilities for reproducibility, metrics, checkpoint IO, and visualization.
- `requirements.txt` – Python dependency list.

## Tips

- Use `--no-pretrained` if you do not want to start from ImageNet weights.
- When no validation set is available, the final epoch checkpoint is saved to `runs/best.pt`.
- Adjust `--num-workers` based on your CPU core count for faster data loading.
