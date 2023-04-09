from torch import nn
import torch
import torchvision


class Model(nn.Module):
    def __init__(self, device):
        super().__init__()

        # Setup the model with pretrained weights and send it to the target device (torchvision v0.13+)
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
        self.model = torchvision.models.efficientnet_b0(weights=weights).to(device)

        # Modify the pretrained models initial Conv2d with in_channels=6
        conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False).to(device)
        # Transfer pretrained weights from Conv2d
        conv1.parameters = self.model.features[0][0].parameters()
        # Replace initial Conv2d from pretrained model with conv1
        self.model.features[0][0] = conv1

        # Recreate the classifier layer and seed it to the target device
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, 
                       inplace=True), 
            nn.Linear(in_features=1280, 
                      out_features=1, # same number of output units as our number of classes
                      bias=True)).to(device)
        
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in self.model.features.parameters():
            param.requires_grad = False
        

    def forward(self, x: torch.Tensor):
        return self.model(x)