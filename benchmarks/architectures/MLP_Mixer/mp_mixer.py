from typing import Optional
import einops
import torch
import torch.nn as nn


class MlpBlock(nn.Module):
    """
    Multilayer percetron.

    Params
    ------
    dims: int
        Input and output dimension of the entire block. Inside of the mixer,
        it will either be equal to `n_patches` or `hidden_dim`

    mlp_dim: int
        Dimension of the hidden layer
    """

    def __init__(self, dim: int, mlp_dim: Optional[int] = None) -> None:
        super().__init__()

        mlp_dim = dim if mlp_dim is None else mlp_dim
        self.linear1 = nn.Linear(dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Param
        -----
        x: torch.Tensor
            Input tensor of shape `(n_samples, n_channels, n_patches)` or
            `(n_samples, n_patches, n_channels)`.

        Returns
        -------
        torch.Tensor
            Output tensor that has exactly the same shape as input `x`.
        """
        x = self.linear1(x)  # (n_samples, *, mlp_dim)
        x = self.activation(x)  # (n_samples, *, mlp_dim)
        x = self.linear2(x)  # (n_samples, *, dim)

        return x


class MixerBlock(nn.Module):
    """
    Mixer block that contains 2 `MlpBlock` and 2 `LayerNorm`.

    Params
    ------
    n_patches: int
        Number of patches the image is split up into.

    hidden_dim: int
        Dimension of patch embeddings.

    tokens_mlo_dim: int
        Hidden dimension for the `MlpBlock` when doing token mixing.

    channels_mlp_dim: int
        Hidden dimension for the `MlpBlock` when doing channel mixing.
    """

    def __init__(
        self,
        *,
        n_patches: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.token_mlp_block = MlpBlock(n_patches, tokens_mlp_dim)
        self.channel_mlp_block = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Param
        -----
        x: torch.Tensor
            Input tensor of shape `(n_samples, n_patches, hidden_dim)`.

        Returns
        -------
        torch.Tensor
            Tensor of same shape as `x`, `(n_samples, n_patches, hidden_dim)`
        """
        y = self.norm1(x)  # (n_samples, n_patches, hidden_dim)
        y = y.permute(0, 2, 1)  # (n_samples, hidden_dim, n_patches)
        y = self.token_mlp_block(y)  # (n_samples, hidden_dim, n_patches)
        y = y.permute(0, 2, 1)  # (n_samples, n_patches, hidden_dim)
        x = x + y  # (n_samples, n_patches, hidden_dim)
        y = self.norm2(x)  # (n_samples, n_patches, hidden_dim)
        y = self.channel_mlp_block(y)
        out = x + y  # (n_samples, n_patches, hidden_dim)
        return out


class MlpMixer(nn.Module):
    """
    MLP Mixer network

    Params
    ------
    in_channels: int
        The color channels of the input image.

    image_size: int
        Height and width (assuming it is a square) of the input image

    patch_size: int
        Height and width (assuming it is a square) of the patches.
        Note that `image_size % patch_size == 0`.

    tokens_mlp_dim: int
        Hidden dimension for the `MlpBlock` when performing token mixing.

    channels_mlp_dim: int
        Hidden dimension for the `MlpBlock` when performing channel mixing.

    n_classes: int
        Number of classes for classification.

    hidden_dim: int
        Dimension of patch embeddings.

    n_blocks: int
        Number of `MixerBlock` in the network.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        hidden_dim: int,
        n_blocks: int,
        n_classes: int,
    ) -> None:
        super().__init__()

        assert (
            image_size % patch_size == 0
        ), "Image size should be divisible by patch size"
        n_patches = (image_size // patch_size) ** 2

        self.patch_embedder = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    n_patches=n_patches,
                    hidden_dim=hidden_dim,
                    tokens_mlp_dim=tokens_mlp_dim,
                    channels_mlp_dim=channels_mlp_dim,
                )
                for _ in range(n_blocks)
            ]
        )

        self.pre_head_norm = nn.LayerNorm(hidden_dim)
        self.head_classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Param
        -----
        x: torch.Tensor
            Input batch of square images of shape
            `(n_samples, n_channels, height, width)`

        Returns
        ------
        torch.Tensor
            Class logits of shape `(n_samples, n_classes)`
        """
        x = self.patch_embedder(x)
        # (n_samples, hidden_dim, n_patches ** (1/2), n_patches ** (1/2))
        x = einops.rearrange(
            x, "n c h w -> n (h w) c"
        )  # (n_samples, n_patches, hidden_dim)
        for mixer_block in self.blocks:
            x = mixer_block(x)  # (n_samples, n_patches, hidden_dim)

        x = self.pre_head_norm(x)  # (n_samples, n_patches, hidden_dim)
        x = x.mean(dim=1)  # (n_samples, hidden_dim)
        y = self.head_classifier(x)  # (n_samples, n_classes)
        return y


if __name__ == "__main__":
    x = torch.randn((100, 3, 224, 224))
    model = MlpMixer(
        in_channels=3,
        image_size=224,
        patch_size=16,
        tokens_mlp_dim=6,
        channels_mlp_dim=6,
        hidden_dim=32,
        n_blocks=4,
        n_classes=10,
    )
    y = model(x)
    print(y.shape)
