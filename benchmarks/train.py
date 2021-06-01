import PIL
import torch
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm
import time
import pandas as pd

from dataset import CIFAR10Data
from architectures import VGG_Net, VGG_types, ResNet50, ViT, EfficientNet


def train(model, optimizer, data_loader, loss_history, device):
    total_samples = len(data_loader.dataset)
    model.to(device)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                "["
                + "{:5}".format(i * len(data))
                + "/"
                + "{:5}".format(total_samples)
                + " ("
                + "{:3.0f}".format(100 * i / len(data_loader))
                + "%)]  Loss: "
                + "{:6.4f}".format(loss.item())
            )
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history, device):
    model.to(device)
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction="sum")
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print(
        "\nAverage test loss: "
        + "{:.4f}".format(avg_loss)
        + "  Accuracy:"
        + "{:5}".format(correct_samples)
        + "/"
        + "{:5}".format(total_samples)
        + " ("
        + "{:4.2f}".format(100.0 * correct_samples / total_samples)
        + "%)\n"
    )


def main():
    N_EPOCHS = 10
    IN_CHANNELS = 3
    NUM_CLASSES = 10
    LR = 0.01

    # Device selection:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model list:
    models = [
        ResNet50(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES),
        VGG_Net(
            arch=VGG_types["VGG16"], in_channels=IN_CHANNELS, num_classes=NUM_CLASSES
        ),
        EfficientNet(version="b0", in_channels=IN_CHANNELS, num_classes=NUM_CLASSES),
        ViT(
            image_size=32,
            patch_size=8,
            num_classes=NUM_CLASSES,
            in_channels=IN_CHANNELS,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=1024,
        ),
    ]

    # Training initialization
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(8, translate=(0.15, 0.15)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor()])

    data = CIFAR10Data(
        transform_train=transform_train, transform_test=transform_test, batch_size=128
    )
    train_loader, val_loader, test_loader = data.initialize_dataset()

    for model in models:
        print(f"Model: {model.name}")

        CHECKPOINT = f"{model.name}_checkpoint.pth"

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        train_loss_history, test_loss_history = [], []

        logs = {
            "model": f"{model.name}",
            "train_loss": None,
            "test_loss": None,
            "time": 0,
        }

        start_time = time.time()
        for epoch in tqdm(range(1, N_EPOCHS + 1)):
            print(f"Epoch {epoch:3}/{N_EPOCHS}")
            train(model, optimizer, train_loader, train_loss_history, device)
            evaluate(model, test_loader, test_loss_history, device)

        total_time = round((time.time() - start_time), 2)
        print(f"Execution time: {(total_time):5.2f} seconds.")

        logs.update(
            {
                "train_loss": train_loss_history,
                "test_loss": test_loss_history,
                "time": total_time,
            }
        )
        dataframe = pd.DataFrame(logs)
        dataframe.to_csv(f"{model.name}_log.csv")

        torch.save(model.state_dict(), CHECKPOINT)


if __name__ == "__main__":
    main()
