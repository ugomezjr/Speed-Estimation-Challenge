from pathlib import Path
from utils.dataset import SpeedEstimations
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os

#NUM_WORKERS = os.cpu_count()

# Setup directories
train_dir = Path("./data/train")
# test_dir = Path("./data/test") 

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

# Instantiate SpeedEstimations dataset
train_data = SpeedEstimations(txt_file=train_dir / "train.txt",
                              root_dir=train_dir,
                              transform=transform)

# Convert images to data loaders
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=32)
                              #shuffle=True,
                              #num_workers=NUM_WORKERS,
                              #pin_memory=True)

images, labels = next(iter(train_dataloader))
print(images.shape, labels.shape)