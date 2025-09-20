# Cat vs. Dog Classifier (PyTorch)

This project trains a PyTorch image classifier to distinguish between cats and dogs. The
pipeline now downloads the [Cats vs Dogs dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
through `torchvision.datasets.CatsVsDogs`, so no manual data preparation is required. By default
the archive is cached in a local `data/` directory; use `--data-dir` to point elsewhere.

Use `--split` (default `0.2`) to reserve a portion of the images for validation. Set `--split 0`
if you want to train on the full dataset without a validation loader.

Mixed-precision (AMP) is enabled automatically on CUDA devices to cut memory usage, and you can
accumulate gradients across batches with `--grad-accum-steps` to simulate larger batch sizes on
GPUs such as the RTX 3060 6GB. Disable AMP with `--no-amp` if you need deterministic FP32 math.

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

Run training and let the script download/cache the dataset automatically. The best checkpoint is
saved to `runs/best.pt`, TensorBoard logs are stored in `runs/tensorboard`, and the normalized
confusion matrix is exported as `runs/confusion_matrix.png`.

```powershell
# Train with the default 80/20 train/val split and AMP enabled (recommended for 6GB GPUs)
python train.py --epochs 15 --batch-size 16 --grad-accum-steps 2

# Train while caching the dataset on another drive and disabling the validation split
python train.py --data-dir D:\\torch_cache --epochs 15 --split 0
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
- `datasets.py` – CatsVsDogs data loading utilities with optional train/validation split.
- `utils.py` – helper utilities for reproducibility, metrics, checkpoint IO, and visualization.
- `requirements.txt` – Python dependency list.

## Tips

- Effective batch size is `batch_size * grad_accum_steps`; tune both values to fit your GPU RAM.
- Use `--no-pretrained` if you do not want to start from ImageNet weights.
- When `--split 0`, the final epoch checkpoint is saved to `runs/best.pt`.
- Adjust `--num-workers` based on your CPU core count for faster data loading.
