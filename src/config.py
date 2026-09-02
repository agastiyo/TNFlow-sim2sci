"""Project paths, device, and shared defaults."""

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.parquet"
SPECTRA_PATH = DATA_DIR / "spectra.parquet"
CHECKPOINT_DIR = PROJECT_ROOT  # checkpoints live at the repo root
BEST_LOSS_CHECKPOINT = CHECKPOINT_DIR / "tnflow_checkpoint.pt"
HISTORY_PATH = PROJECT_ROOT / "training_history.json"  # per-epoch train/val/OOD curves (for figures)


def get_device() -> torch.device:
    """CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
