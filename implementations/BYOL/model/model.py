import torch
import torch.nn as nn
from torchvision import models


class MlpHead(nn.Module):
    def __init__(self, in_channels, hidden_size, projection_size):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.head(x)


class ResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if kwargs["name"] == "resnet18":
            resnet = models.resnet18(pretrained=False)
        elif kwargs["name"] == "resnet50":
            resnet = models.resnet50(pretrained=False)

        self.encoder = nn.Sequential(*list(resnet.children()[:-1]))
        self.projection = MlpHead(
            in_channels=resnet.fc.in_features, **kwargs["projection_head"]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], x.shape[1])
        out = self.projection(x)
        return out
