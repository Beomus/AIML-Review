from math import ceil
import torch
from torch import nn


base_model = [
    # List[expand ratio, channels, repeats, stride, kernel_size]
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple(phi_value, resolution, drop_rate)
    # alpha, beta, gamma, depth = alpha ** phi
    # alpha: for depth of the model
    # beta: for width of the model
    # gamma: for resolution
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,  # depth-wise convolution
            # if groups=1, then it is default conv
            # if groups=in_channels then it is depth-wise conv
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # SiLU <-> Swish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, reduced_dim: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # CxHxW -> Cx1x1
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        expand_ratio: int,
        reduction: int = 4,  # for SqueezeExcitation
        survival_prob: float = 0.8,  # for Stochastic depth
    ) -> None:
        super().__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = ConvBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1
            )

        self.conv = nn.Sequential(
            ConvBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x: torch.Tensor) -> torch.Tensor:
        # only for training, not for testing
        if not self.training:
            return x

        binary_tensor = (
            torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        )
        # dividing by survival_prob to somewhat preserve mean and std values for BN
        # multiply by `binary_tensor` to perform dropout of layers
        out = torch.div(x, self.survival_prob) * binary_tensor
        return out

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            # adding residual with stochastic depth if using residual
            x = self.conv(x)
            x = self.stochastic_depth(x) + inputs
            return x
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version: str, in_channels: int, num_classes: int) -> None:
        super().__init__()
        width_factor, depth_factor, dropout_rate = self._calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.in_channels = in_channels
        self.name = "EfficientNet"
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self._create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(last_channels, num_classes)
        )

    def _calculate_factors(self, version: str, alpha: float = 1.2, beta: float = 1.1):
        phi, resolution, drop_rate = phi_values[version]
        depth_factor = alpha**phi
        width_factor = beta**phi
        return width_factor, depth_factor, drop_rate

    def _create_features(
        self, width_factor: int, depth_factor: int, last_channels: int
    ) -> nn.Sequential:
        channels = int(32 * width_factor)
        features = [
            ConvBlock(self.in_channels, channels, kernel_size=3, stride=2, padding=1)
        ]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        # if k=1:pad=0, if k=3:pad=1, if k=5:pad=2
                    )
                )
                in_channels = out_channels

        features.append(
            ConvBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x


def test():
    version = "b0"
    phi, res, drop_rate = phi_values[version]
    num_examples, num_classes, in_channels = 1, 10, 3
    x = torch.randn((num_examples, in_channels, res, res))
    model = EfficientNet(
        version=version, in_channels=in_channels, num_classes=num_classes
    )
    y = model(x)
    print(y.shape)  # (num_examples, num_classes)
    print(y)


if __name__ == "__main__":
    test()
