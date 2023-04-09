from dataset import SpeedEstimations
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from pathlib import Path
import os

def create_dataloaders(data_dir: Path,
                       labels_dir: Path,
                       train_split: float=0.8,
                       valid_split: float=0.2, 
                       transform: transforms.Compose=None,
                       batch_size: int=32,
                       num_workers: int=os.cpu_count()):

    # Create(s) an instance of the SpeedEstimations dataset
    dataset = SpeedEstimations(txt_file=labels_dir,
                               root_dir=data_dir, 
                               transform=transform)
    
    dataset_size = len(dataset)

    train_index = int( dataset_size * train_split )
    valid_index = int( dataset_size * train_split )

    train_indices = list(range(train_index))
    valid_indices = list(range(train_index, train_index + valid_index))

    train_data = Subset(dataset=dataset, indices=train_indices)
    valid_data = Subset(dataset=dataset, indices=valid_indices)

    # Turn frames into dataloaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True)
    valid_dataloader = DataLoader(dataset=valid_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True)
    
    return train_dataloader, valid_dataloader 