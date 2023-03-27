import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


class LinearAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def main():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )
    mnist_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=mnist_data, batch_size=64, shuffle=True
    )

    linear_model = LinearAE()
    linear_model = linear_model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = 10
    linear_outputs = []
    print("Start training for LinearAE")
    for epoch in range(epochs):
        for img, _ in data_loader:
            img = img.reshape(-1, 28 * 28)
            img = img.to(device)
            recon = linear_model(img)
            loss = criterion(recon, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch} | Loss: {loss.item():.4f}")
        linear_outputs.append((epoch, img, recon))

    conv_model = ConvAE()
    conv_model = conv_model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(conv_model.parameters(), lr=1e-3, weight_decay=1e-5)
    conv_outputs = []
    print("Start training for LinearAE")
    for epoch in range(epochs):
        for img, _ in data_loader:
            img = img.to(device)
            recon = conv_model(img)
            loss = criterion(recon, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch} | Loss: {loss.item():.4f}")
        conv_outputs.append((epoch, img, recon))


if __name__ == "__main__":
    main()
