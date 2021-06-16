from typing import List, Optional, Type, Union
import torch
from torch import nn


class BasicBlock(nn.Module):
    """
    Basic block for ResNet34 and ResNet18
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        identity_downsample: Optional[nn.Module] = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, stride=stride, kernel_size=3, padding=1
        )  # conv3x3
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, stride=1, kernel_size=3, padding=1
        )  # conv3x3
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity_downsample = identity_downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet50 and above.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        identity_downsample: Optional[nn.Module] = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )  # conv1x1
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )  # conv3x3
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )  # conv1x1
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.identity_sample = identity_downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.identity_sample is not None:
            identity = self.identity_sample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    """
    ResNet model architecture.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.name = "ResNet"
        self.in_channels = 64
        out_channels = [64, 128, 256, 512]
        self.conv1 = nn.Conv2d(
            in_channels,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layers(block, layers[0], out_channels[0], stride=1)
        self.layer2 = self._make_layers(block, layers[1], out_channels[1], stride=2)
        self.layer3 = self._make_layers(block, layers[2], out_channels[2], stride=2)
        self.layer4 = self._make_layers(block, layers[3], out_channels[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[-1] * block.expansion, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)  # x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layers(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: int,
        out_channels: int,
        stride: int = 1,
    ) -> nn.Sequential:
        identity_downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # change the channels out_channels: 64
        layers.append(
            block(self.in_channels, out_channels, identity_downsample, stride)
        )

        # change in_channels to match
        self.in_channels = out_channels * block.expansion

        # skip the first block since we already initialized it
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
            # in_channels: 256 -> out_channels: 64
            # in the bottleneck block, the final out_channels
            # will be multiplied by 4, 64 -> 256 again
            # at the end of this loop, out_channels would be 256 again.

        return nn.Sequential(*layers)


def ResNet18(in_channels=3, num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes)


def ResNet34(in_channels=3, num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channels, num_classes)


def ResNet50(in_channels=3, num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels, num_classes)


def ResNet101(in_channels=3, num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], in_channels, num_classes)


def ResNet152(in_channels=3, num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], in_channels, num_classes)


def test():
    nets = [ResNet18(), ResNet34(), ResNet50(), ResNet101(), ResNet152()]
    x = torch.randn(2, 3, 224, 224)  # batch size of 3 images of 3 channels
    for net in nets:
        y = net(x)
        print(y.shape)


if __name__ == "__main__":
    test()
