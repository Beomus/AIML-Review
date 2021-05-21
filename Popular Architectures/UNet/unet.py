from typing import List
import torch
from torch import nn
from torchvision.transforms import functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_chans: int, out_chans: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_chans, out_chans, kernel_size=3, stride=1, padding=1
            ),  # same conv
            nn.BatchNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self, in_chans: int, out_chans: int, features: List[int] = [64, 128, 256, 512]
    ) -> None:
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_chans=in_chans, out_chans=feature))
            in_chans = feature

        # Up part
        for feature in reversed(features):
            # doubling feature for concat
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_chans, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)  # go through DoubleConv
            skip_connections.append(x)  # save skip connection
            x = self.pool(x)  # go through pooling

        # go through Bottleneck
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]  # reverse order to concat

        for idx in range(0, len(self.ups), 2):
            # step of 2 since there are 8 layers in self.ups
            # ConvTranpose2d @ even idx
            # DoubleConv @ odd idx
            x = self.ups[idx](x)  # go up through ConvTranpose2d
            skip_connection = skip_connections[idx // 2]

            # sanity check for concat
            if x.shape != skip_connection.shape:
                # TODO: there are better ways to do this
                # print(f"resizing from {x.shape[2:]} to {skip_connection.shape[2:]}")
                # resizing x since x is always smaller or equal to skip connection
                x = TF.resize(x, size=skip_connection.shape[2:])  # skipping b, c

            # concat along the channel dim (dim=1 -> b, c, h ,w)
            concat_skip = torch.cat((skip_connection, x), dim=1)

            # go through doubleconv after concatenating
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        return x


def test():
    x = torch.rand((3, 3, 161, 161))
    print(f"x shape: {x.shape}")
    model = UNet(in_chans=3, out_chans=3)
    preds = model(x)
    print(f"preds shape: {preds.shape}")
    assert preds.shape == x.shape, "Shapes are not equal."


if __name__ == "__main__":
    test()
