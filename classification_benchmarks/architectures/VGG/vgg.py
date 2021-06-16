from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG_Net(nn.Module):
    def __init__(self, arch: List, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.name = "VGG"
        self.conv_layers = self.build_conv(arch)
        self.fc = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),  # 224 -> 5 maxpool = 224/(2**5) = 7
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def build_conv(self, arch: List) -> nn.Sequential:
        layers = []
        in_channels = self.in_channels

        for i in arch:
            if type(i) == int:
                out_channels = i

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(i),
                    nn.ReLU(),
                ]

                in_channels = i
            elif i == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)


def test():
    x = torch.randn(1, 3, 224, 224)
    model = VGG_Net(arch=VGG_types["VGG16"], in_channels=3, num_classes=1000)
    print(model(x).shape)


if __name__ == "__main__":
    test()
