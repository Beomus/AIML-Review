from einops import rearrange
import time
import PIL
import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch import nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs)


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)

        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads  # type: ignore
        qkv = self.to_qkv(x)  # gets q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(
            qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=h
        )  # split into multihead attentions

        dots = (
            torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        )  # scale to avoid gradient issues

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert (
                mask.shpape[-1] == dots.shape[-1]
            ), f"Mask has incorrect dimension (dots dim {dots.shape[-1]})"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float("-inf"))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)  # product of v and attention
        out = rearrange(
            out, "b h n d -> b n (h d)"
        )  # concat heads into one maxtrix read for next encoder
        out = self.nn1(out)
        out = self.do1(out)

        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            LayerNormalize(
                                dim, Attention(dim, heads=heads, dropout=dropout)
                            )
                        ),
                        Residual(
                            LayerNormalize(dim, MLPBlock(dim, mlp_dim, dropout=dropout))
                        ),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        in_channels=3,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), f"Image dimensions ({image_size}) must be divisible by patch size ({patch_size})"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.name = "ViT"
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_patches + 1), dim))
        torch.nn.init.normal_(
            self.pos_embedding, std=0.02
        )  # initialized based on the paper
        self.patch_conv = nn.Conv2d(
            3, dim, patch_size, stride=patch_size
        )  # equivalent to x matmul E, with E = embedded matrix

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(
            dim, num_classes
        )  # if finetuning, just use a linear layer without futher hidden layers
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

        """
        # Using addition layers when training on large datasets
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(mlp_dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias)
        self.do2 = nn.Dropout(dropout)
        """

    def forward(self, image, mask=None):
        p = self.patch_size
        # each of 64 vectors is linearly transformed with a FFN equiv to E matmul
        x = self.patch_conv(image)
        # 64 vectors in rows representing 64 patches, each 64 * 3 long
        x = rearrange(x, "b c h w -> b (h w) c")
        cls_tokens = self.cls_token.expand(image.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        # add more layers here according to initialization

        return x


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
    BATCH_SIZE_TRAIN = 1000
    BATCH_SIZE_TEST = 1000
    DL_PATH = "data"
    N_EPOCHS = 300
    CHECKPOINT = "my_checkpoint.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
            transforms.RandomAffine(8, translate=(0.15, 0.15)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = datasets.CIFAR100(
        DL_PATH, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR100(
        DL_PATH, train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False
    )

    model = ViT(
        image_size=32,
        patch_size=8,
        num_classes=100,
        in_channels=1,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=1024,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loss_history, test_loss_history = [], []
    for epoch in range(1, N_EPOCHS + 1):
        print(f"Epoch {epoch:3}/{N_EPOCHS}")
        start_time = time.time()
        train(model, optimizer, train_loader, train_loss_history, device)
        print(f"Execution time: {(time.time() - start_time):5.2f} seconds.")
        evaluate(model, test_loader, test_loss_history, device)

    torch.save(model.state_dict(), CHECKPOINT)


if __name__ == "__main__":
    main()
