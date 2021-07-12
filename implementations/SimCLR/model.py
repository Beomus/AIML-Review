import torch.nn as nn
import torchvision.models as models


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super().__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
        }

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.backbone.fc
        )

    def _get_basemodel(self, base_model):
        return self.resnet_dict[base_model]

    def forward(self, x):
        return self.backbone(x)
