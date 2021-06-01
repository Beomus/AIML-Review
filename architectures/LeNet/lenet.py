import torch
from torch import nn


class LeNet(nn.Module):
    # 1x32x32 input -> (5x5),s=1,p=0 -> avgpool,s=2,p=0 -> (5x5),s=1,p=0
    # -> avgpool,s=2,p=0, (5x5) -> fc 120 -> fc 84 -> fc 10
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0
        )
        self.linear1 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.classifier(x)
        return x


def test():
    x = torch.randn(64, 1, 32, 32)
    model = LeNet()
    print(model(x).shape)


if __name__ == "__main__":
    test()
