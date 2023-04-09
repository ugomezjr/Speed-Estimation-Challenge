from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchvision

class SpeedEstimations(Dataset):
  """Speed Challenge dataset."""
  def __init__(self, txt_file, root_dir, transform=None):
    """
    Arguments:
    txt_file (string): Path to the text file with speed estimations. 
    root_dir (string): Directory with all the images.
    transform (callable, optional): Optional transform to be applied
      on an image. 
    """
    self.speed_estimates = pd.read_csv(txt_file, header=None)
    self.root_dir = Path(root_dir)
    self.transform = transform if transform else torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


  def __len__(self):
    return len(self.speed_estimates)


  def __getitem__(self, idx: int):
    frames = (self.get_frame(idx), self.get_frame(idx+1))
    estimate = self.speed_estimates.iloc[idx+1, 0]

    frames = (self.transform(frames[0]),
              self.transform(frames[1]))

    frames = torch.cat(frames, dim=0)

    return frames, torch.tensor((1, estimate))


  def get_frame(self, i: int):
    return Image.open(self.root_dir / f"frame{i}.png")