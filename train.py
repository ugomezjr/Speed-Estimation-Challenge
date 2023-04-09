from pathlib import Path
from utils.dataset import SpeedEstimations
from utils.model import Model
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os

import torch.nn.functional as F
import torch.optim as optim

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


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred.type(torch.float32), y.type(torch.float32))
        print(loss)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


if __name__ == "__main__":
    model = Model(device=device)
    optimizer = optim.Adam(model.parameters())
    loss, acc = train_step(model=model,
                           dataloader=train_dataloader,
                           loss_fn=F.mse_loss,
                           optimizer=optimizer)
    print(loss, acc)