import os
import torch
from torch.utils import data
from torchvision import datasets
import yaml

from transforms import MultiViewDataInjector, get_simclr_data_transforms
from model import MlpHead, ResNet
from trainer import BYOLTrainer

print(torch.__version__)
torch.manual_seed(0)


def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_transforms = get_simclr_data_transforms(**config["data_transforms"])
    train_dataset = datasets.STL10(
        "./data/",
        split="train+unlabeled",
        download=True,
        transform=MultiViewDataInjector([data_transforms, data_transforms]),
    )

    # online network
    online_network = ResNet(**config["network"]).to(device)
    pretrained_folder = config["network"]["fine_tune_from"]

    # load pre-trained model
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join(
                "./runs", pretrained_folder, "checkpoints"
            )

            load_params = torch.load(
                os.path.join(os.path.join(checkpoints_folder, "model.pth")),
                map_location=torch.device(device),
            )

            online_network.load_state_dict(load_params["online_network_state_dict"])

        except FileNotFoundError:
            print("Pre-trained weights not found, starting training from scratch.")

    predictor = MlpHead(
        in_channels=online_network.projection.head[-1].out_features,
        **config["network"]["projection_head"],
    ).to(device)

    # target encoder
    target_network = ResNet(**config["network"]).to(device)
    optimizer = torch.optim.SGD(
        list(online_network.parameters()) + list(predictor.parameters()),
        **config["optimizer"]["params"],
    )

    trainer = BYOLTrainer(
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer,
        predictor=predictor,
        device=device,
        **config["trainer"],
    )

    trainer.train(train_dataset)


if __name__ == "__main__":
    main()
