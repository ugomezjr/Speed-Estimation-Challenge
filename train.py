from pathlib import Path
from utils.dataset import SpeedEstimations
from utils.model import Model
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from utils.data_setup import create_dataloaders
import torch
import os

import torch.nn.functional as F
import torch.optim as optim

#NUM_WORKERS = os.cpu_count()

# Setup directories
data_dir = Path("./data/train")
labels_dir = data_dir / "train.txt"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, valid_dataloader = create_dataloaders(data_dir=data_dir,
                                                        labels_dir=labels_dir)


    