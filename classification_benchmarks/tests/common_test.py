from classification_benchmarks.architectures.MLP_Mixer.mlp_mixer import MlpMixer
import torch
import unittest

from ..architectures import *


class TestArchs(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 3
        self.in_channels = 3
        self.num_classes = 100
        self.x = torch.randn(self.batch_size, self.in_channels, 224, 224)
        # print(f"\nInitialize x: {self.x.shape=}")

    def test_ResNet(self):
        model = ResNet50(self.in_channels, self.num_classes)
        y = model(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.num_classes))

    def test_EfficientNet(self):
        version = "b0"
        model = EfficientNet(
            version=version, in_channels=self.in_channels, num_classes=self.num_classes
        )
        y = model(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.num_classes))

    def test_MlpMixer(self):
        model = MlpMixer(
            in_channels=self.in_channels,
            image_size=224,
            patch_size=16,
            tokens_mlp_dim=6,
            channels_mlp_dim=6,
            hidden_dim=32,
            n_blocks=4,
            n_classes=self.num_classes,
        )
        y = model(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.num_classes))

    def test_VGG(self):
        arch = VGG_types["VGG16"]
        model = VGG_Net(
            arch=arch, in_channels=self.in_channels, num_classes=self.num_classes
        )
        y = model(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.num_classes))

    def test_ViT(self):
        model = ViT(
            in_channels=self.in_channels,
            image_size=224,
            patch_size=8,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=1024,
            num_classes=self.num_classes,
        )
        y = model(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.num_classes))


if "__name__" == "__main__":
    unittest.main()
